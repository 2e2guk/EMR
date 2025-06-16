import torch
import argparse
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu

# 우리가 직접 만든 클래스들을 import
from lavis.models.blip2_models.blip2_llama_instruct_malmm import Blip2LlamaInstruct_MALMM
from lavis.datasets.datasets.visdial_dataset import VisDialDataset
from lavis.processors.blip_processors import BlipCaptionProcessor


def evaluate(model, dataset, device, batch_size=16, max_new_tokens=30 , num_beams=5):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=dataset.collater,
    )

    model.eval()
    model.to(device)

    references = []
    candidates = []

    print("Starting evaluation...")
    with torch.no_grad():
        for samples in tqdm(data_loader, desc="Evaluating"):
            samples["image_embedding"] = samples["image_embedding"].to(device)
            samples["text_embedding"] = samples["text_embedding"].to(device)
            samples["turn_id"] = samples["turn_id"].to(device)

            # [핵심 수정] generate 함수에 길이 관련 인자를 직접 전달합니다.
            generated_captions = model.generate(
                samples,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

            # ... (이하 코드는 동일)
            ground_truth_captions = samples["text_output"]
            candidates.extend(generated_captions)
            references.extend(ground_truth_captions)

    references_tokenized = [[ref.split()] for ref in references]
    candidates_tokenized = [cand.split() for cand in candidates]

    bleu4 = corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n--- Evaluation Complete ---")
    print(f"Total samples evaluated: {len(candidates)}")
    print(f"BLEU-4 Score: {bleu4 * 100:.2f}")


if __name__ == "__main__":
    # ... (parser 및 모델 로딩 부분은 동일) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--llm_model_path", required=True, help="Path to the base LLM model.")
    args = parser.parse_args()

    print("Initializing model architecture...")
    model = Blip2LlamaInstruct_MALMM(
        llm_model=args.llm_model_path
    )

    print(f"Loading checkpoint weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)

    print("Loading VisDial validation dataset...")
    text_processor = BlipCaptionProcessor()
    val_dataset = VisDialDataset(
        vis_processor=None,
        text_processor=text_processor,
        vis_root="/home/leegw/visdial_imagebind_embeddings",
        ann_paths=["/home/leegw/dataset/visdial/visdial_1.0_val.json"]
    )


    evaluate(model, val_dataset, device="cuda", max_new_tokens=30, num_beams=5)