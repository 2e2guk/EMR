"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, memory_bank_compress, LayerNorm

from PIL import Image
from torchvision import transforms


#from LAVIS_backup.tests.models.test_pnp_vqa import device


#from lavis.projects.xinstructblip.discrn.caption_baseline.predict_audio import config


@registry.register_model("blip2_llama_instruct_malmm")
class Blip2LlamaInstruct_MALMM(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "llama1b": "configs/models/blip2/blip2_instruct_llama1b.yaml",
    }

    # imagebind 연동 init 함수.
    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="",
            freeze_llm=True,
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
            memory_bank_length=0,
            num_frames=0,
            max_num_frames=120,
            cross_attention_freq=2,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"

        # =============================================================
        # [최종 수정] Vision Encoder -> FC Layer로 교체
        # =============================================================

        # 1. 토크나이저는 그대로 초기화.
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # 2. ImageBind의 vision(1024)과 text(1024) 임베딩을 합쳐(2048)
        #    Q-Former가 받을 차원(1024)으로 변환하는 새로운 FC 레이어를 정의.
        self.imagebind_fc = nn.Linear(1024 + 1024, 1024)
        vision_width = 1024  # FC Layer의 출력 크기, Q-Former의 입력 크기가 된다.

        # 3. Layer Normalization은 FC Layer의 출력에 적용하기 위해 그대로 둔다.
        self.ln_vision = LayerNorm(vision_width)

        # 4. Q-Former를 초기화한다.
        #    vision_width가 FC Layer의 출력 차원인 1024로 전달된다.
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            vision_width,
            cross_attention_freq=cross_attention_freq,
            memory_bank_length=memory_bank_length,
            num_frames=num_frames,
        )

        # =============================================================

        # Q-Former의 후처리 부분은 기존과 동일.
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # LLM 로딩 부분. (4-bit 양자화 적용)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=True, truncation_side="left")

        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto"
        )

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # LLM 프로젝션 레이어도 Q-Former의 출력(768)을 기준으로 설정했던 것을 그대로 유지.
        # (blip2.py에서 Q-Former의 hidden_size를 768로 설정했기 때문)
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
        self.turn_pe = nn.Embedding(11, 1024)

        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        # MemoryBank 관련 설정도 그대로 유지합니다.
        self.qformer_text_input = qformer_text_input
        self.num_query_token = num_query_token
        self.memory_bank_length = memory_bank_length
        self.use_memory_bank = True if memory_bank_length > 0 else False
        self.num_frames = num_frames
        self.visual_memory_bank = None
        self.image_pe = nn.Embedding(max_num_frames, 1024)  # ImageBind 임베딩 차원에 맞춤
        nn.init.constant_(self.image_pe.weight, 0.0)

        # 이 부분은 데이터셋에서 처리하므로 모델에서는 직접 사용하지 않을 수 있습니다.
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        ])

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # 1. 데이터셋에서 사전 추출된 임베딩과 턴 ID를 받습니다.
        image_embedding = samples["image_embedding"].to(self.device)
        text_embedding = samples["text_embedding"].to(self.device)
        turn_ids = samples["turn_id"].to(self.device)

        # 2. 내용 임베딩(Content Embedding) 생성
        # vision, text 임베딩을 합치고 FC Layer를 통과시킵니다.
        combined_embedding = torch.cat([image_embedding, text_embedding], dim=1)
        content_embedding = self.imagebind_fc(combined_embedding)

        # 3. 순서 임베딩(Positional Embedding) 생성 및 결합
        # 턴 ID를 이용해 순서 임베딩을 조회하고, 내용 임베딩에 더해줍니다.
        turn_positional_embeddings = self.turn_pe(turn_ids)
        final_embedding = content_embedding + turn_positional_embeddings

        # 4. Q-Former 입력 형식에 맞게 최종 임베딩을 가공합니다.
        # (batch_size, 1024) -> (batch_size, 1, 1024)
        embedding_for_qformer = final_embedding.unsqueeze(1)
        image_embeds = self.ln_vision(embedding_for_qformer)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # 5. Q-Former에 텍스트(질문)와 최종 임베딩 특징을 전달합니다.
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 6. Q-Former 출력을 LLM에 전달하기 위해 프로젝션합니다.
        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # 7. LLM 학습을 위한 입력 및 레이블을 준비합니다.
        self.llm_tokenizer.padding_side = "right"
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'], return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len
        ).to(image_embeds.device)
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_output_txt_len
        ).to(image_embeds.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids, text_input_tokens.attention_mask,
            text_output_tokens.input_ids, text_output_tokens.attention_mask
        )

        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image_embeds.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        # 8. 최종 LLM 순전파 및 손실 계산
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_new_tokens=30,  # max_length, min_length 대신 사용
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # 1. 임베딩 처리 로직
        image_embedding = samples["image_embedding"].to(self.device)
        text_embedding = samples["text_embedding"].to(self.device)
        turn_ids = samples["turn_id"].to(self.device)

        combined_embedding = torch.cat([image_embedding, text_embedding], dim=1)
        content_embedding = self.imagebind_fc(combined_embedding)
        turn_positional_embeddings = self.turn_pe(turn_ids)
        final_embedding = content_embedding + turn_positional_embeddings
        visual_embedding_for_qformer = final_embedding.unsqueeze(1)
        image_embeds = self.ln_vision(visual_embedding_for_qformer)

        # 2. Q-Former 및 LLM 입력 준비
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        text_Qformer = self.tokenizer(
            samples["text_input"], padding='longest', truncation=True,
            max_length=self.max_txt_len, return_tensors="pt"
        ).to(image_embeds.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)

        llm_tokens = self.llm_tokenizer(
            samples["text_input"], padding="longest", return_tensors="pt"
        ).to(self.device)

        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

        # 3. [핵심] LLM 답변 생성 시 max_new_tokens 사용
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,  # <--- 수정된 인자 전달
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in
                                        enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_captions=num_beams,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id,
                                                              -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    # 추가
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        # self.imagebind_fc 의 parameter만 반환
        params_to_train = list(self.imagebind_fc.parameters()) + list(self.turn_pe.parameters())
        return params_to_train

    @classmethod
    def from_config(cls, cfg):
        # 1. 설정 값들을 딕셔너리로 먼저 추출합니다.
        model_cfg = cfg.get("model", {})
        init_params = {
            "vit_model": model_cfg.get("vit_model", "eva_clip_g"),
            "img_size": model_cfg.get("image_size", 224),
            "drop_path_rate": model_cfg.get("drop_path_rate", 0),
            "use_grad_checkpoint": model_cfg.get("use_grad_checkpoint", False),
            "vit_precision": model_cfg.get("vit_precision", "fp16"),
            "freeze_vit": model_cfg.get("freeze_vit", True),
            "num_query_token": model_cfg.get("num_query_token", 32),
            "llm_model": model_cfg.get("llm_model"),
            "freeze_llm": model_cfg.get("freeze_llm", True),
            "prompt": model_cfg.get("prompt", ""),
            "max_txt_len": model_cfg.get("max_txt_len", 128),
            "max_output_txt_len": model_cfg.get("max_output_txt_len", 256),
            "apply_lemmatizer": cfg.get("apply_lemmatizer", False),
            "qformer_text_input": cfg.get("qformer_text_input", True),
            "cross_attention_freq": model_cfg.get("cross_attention_freq", 2),
        }

        # 2. 추출한 파라미터로 모델 인스턴스를 생성합니다.
        model = cls(**init_params)

        # 3. Gradient Checkpointing을 활성화합니다.
        model.llm_model.gradient_checkpointing_enable()

        # 4. 'finetuned' 경로의 체크포인트를 직접 불러옵니다.
        ckpt_path = model_cfg.get("finetuned", "")
        if ckpt_path:
            print(f"Loading finetuned checkpoint from: {ckpt_path}")
            # PyTorch 2.6+ 호환성을 위해 weights_only=False 추가
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # 체크포인트는 'model' 키 안에 가중치를 가지고 있습니다.
            #state_dict = ckpt.get('model', ckpt)
            state_dict = ckpt.get('model')

            # strict=False로 설정하여, 현재 모델에 없는 키(예: 옛날 visual_encoder)는 무시하고
            # 일치하는 키(예: 우리가 학습시킨 imagebind_fc)의 가중치만 선택적으로 불러옵니다.
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loading message: {msg}")

        return model