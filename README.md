# (2024CVPR) MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding
### [Project Page](https://boheumd.github.io/MA-LMM/) | [Paper](https://arxiv.org/abs/2404.05726)
The official repository of our paper "**MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding**".

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ma-lmm-memory-augmented-large-multimodal/video-classification-on-breakfast)](https://paperswithcode.com/sota/video-classification-on-breakfast?p=ma-lmm-memory-augmented-large-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ma-lmm-memory-augmented-large-multimodal/video-classification-on-coin-1)](https://paperswithcode.com/sota/video-classification-on-coin-1?p=ma-lmm-memory-augmented-large-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ma-lmm-memory-augmented-large-multimodal/visual-question-answering-on-msvd-qa-1)](https://paperswithcode.com/sota/visual-question-answering-on-msvd-qa-1?p=ma-lmm-memory-augmented-large-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ma-lmm-memory-augmented-large-multimodal/video-question-answering-on-msrvtt-qa)](https://paperswithcode.com/sota/video-question-answering-on-msrvtt-qa?p=ma-lmm-memory-augmented-large-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ma-lmm-memory-augmented-large-multimodal/video-captioning-on-youcook2)](https://paperswithcode.com/sota/video-captioning-on-youcook2?p=ma-lmm-memory-augmented-large-multimodal)


<p align="center">
<img src="figs/teaser.png" alt="teaser" width="60%">
</p>


## Model Overview
<p align="center">
<img src="figs/architecture.png" alt="model" width="80%">
</p>

## Demo
You can explore our demo by running `demo.ipynb`. This demonstration illustrates how our MA-LMM serves as a plug-and-play module that can be integrated into InstructBLIP seamlessly, requiring no fine-tuning for zero-shot evaluation.

## Requirements

You can install the conda environment by running:
```bash
git clone https://github.com/boheumd/MA-LMM.git
cd MA-LMM
pip install -e .
```

If you are running the code on Apple Silicon, you need to use `eva-decord` instead of `decord`. Here is the modification in the `requirements.txt` file you should do:

```text
contexttimer
eva-decord
einops>=0.4.1
fairscale==0.4.4
...
```

**Before running `pip install -e .`, ensure you have the correct requirements.**

## Dataset
For the long-term video understanding task, we conduct experiments including ([LVU](https://github.com/chaoyuaw/lvu)) and two standard video summarization datasets ([Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), [COIN](https://coin-dataset.github.io/)).

For the video question answering task, we conduct experiments including [MSRVTT](https://github.com/xudejing/video-question-answering), [MSVD](https://github.com/xudejing/video-question-answering), and [ActivityNet](https://github.com/MILVLG/activitynet-qa).
For the video captioning task, we also conduct experiments on [Youcook2](http://youcook2.eecs.umich.edu/) dataset.

You can download videos for each dataset through the script provided here (lavis/datasets/download_scripts). For LVU/Breakfast/COIN datasets, please download the original videos through the official link provided above.

Then extract video frames of each video with fps=10. Example preprocess code is provided here [extract_frames.py](https://github.com/boheumd/MA-LMM/blob/main/data/extract_frames.py).
Since different FFMPEG versions are used, the actual extracted frame lengths can be slightly inconsistent. You may need to update the actual frame_length for each video in the annotation file.
   ```
    ├── data
        └── activitynet
            ├── annotation
            ├── frames
            ├── videos
        └── breakfast
            ├── annotation
            ├── frames
            ├── videos
        └── coin
            ├── annotation
            ├── frames
            ├── videos
        └── lvu
            ├── annotation
            ├── frames
            ├── videos
        └── msrvtt
            ├── annotation
            ├── frames
            ├── videos
        └── msvd
            ├── annotation
            ├── frames
            ├── videos
        └── youcook2
            ├── annotation
            ├── frames
            ├── videos
   ```



## Running

### Download Pre-trained LLM
We use Vicuna-v1.1 as our pre-trained LLM weights, you can download from this [link](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) as arrange in this format.
   ```
   ├── llm
        ├── vicuna-7b
        ├── vicuna-13b
   ```
### Finetuning on Downstreaming Tasks
Our model leverages pre-trained weights from [InstructBlip](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip), which was only pre-trained on image-text pairs. Our training process occurred on four A100 GPUs. If you would like to fine-tune the model for various video datasets, please run the following command:
```bash
bash run_scripts/${dataset}/train.sh
```

#### LVU dataset
```bash
    # Please choose the task from the following list
    # ['director', 'genre', 'relationship', 'scene', 'way_speaking', 'writer', 'year']
    datasets.lvu_cls.task ${task}
```

### Testing
We also provided finetuned checkpoints for each video dataset. Please download the [saved_model.tar](https://drive.google.com/file/d/1mq6fg69Ofm32-1HjEunoFtPg8ymAIcOp/view?usp=sharing) and unzip it. 
For the test script corresponding to each dataset, provide the path to the extracted checkpoint to execute the evaluation.
```bash
bash run_scripts/${dataset}/test.sh ${checkpoint_path}
```

### Zero-shot Evaluation
Our model can also leverage pre-trained weights from [InstructBlip](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) without any finetuning to conduct zero-shot evaluation on video datasets.
```bash
bash run_scripts/${dataset}/test.sh
```


### Hyper-parameters
One important hyper-parameters memory_bank_length, please change that in the training script on different datasets.
```bash
    # pre-defined length of the memory bank
    model.memory_bank_length ${value}
    # value=0 means without using the memory bank
```

### Memory Bank Compression Code
The core algorithm for the memory bank compression algorithm is [here](https://github.com/boheumd/MA-LMM/blob/main/lavis/models/blip2_models/blip2.py#L352).

## Citation
If you find our code or our paper useful for your research, please **[★star]** this repo and **[cite]** the following paper:

```latex
@inproceedings{he2024malmm,
  title = {MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding},
  author    = {He, Bo and Li, Hengduo and Jang, Young Kyun and Jia, Menglin and Cao, Xuefei and Shah, Ashish and Shrivastava, Abhinav and Lim, Ser-Nam},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```


## Acknowledgement
We referenced the repo below for the code
- [LAVIS](https://github.com/salesforce/LAVIS)



## Additional by 2e2guk

### MA-LMM with ImageBind: Multimodal Memory & Dialogue System

#### 1. 프로젝트 개요 (Overview)
본 프로젝트는 Vision, Audio 등 다양한 모달리티를 하나의 임베딩 공간에 결합하는 ImageBind 모델을 기반으로, 장기 기억 및 추론 능력을 갖춘 멀티모달 기억 시스템을 개발하는 것을 목표로 합니다.  
기존 대형언어모델(LLM)의 한계인 장기 기억 부족과 추론 오류를 극복하기 위해, **MA-LMM (Memory-Augmented Large Multimodal Model)**의 아키텍처를 차용하고, 여기에 ImageBind의 강력한 멀티모달 인코딩 능력을 결합했습니다.

#### 2. 최종 모델 아키텍처
본 프로젝트에서 구현된 모델은 다음과 같은 독자적인 파이프라인 아키텍처를 가집니다.

##### 입력 인코더 (Frozen)
- **Image Encoder**: ImageBind-Huge의 Vision Encoder (출력: 1024차원)  
- **Text Encoder**: ImageBind-Huge의 Text Encoder (출력: 1024차원)

##### 연결 모듈 (Trainable)
- **FC Layer (imagebind_fc)**: ImageBind의 Vision, Text 임베딩을 결합한 2048차원 벡터를 받아, Q-Former가 처리할 수 있는 1024차원 벡터로 변환하는 `nn.Linear(2048, 1024)` 레이어  
- **순서 임베딩 (turn_pe)**: Visual Dialog의 턴(turn) 순서 정보(0~10)를 학습하기 위한 `nn.Embedding(11, 1024)` 레이어

##### 브릿지 모듈 (Frozen)
- **Q-Former**: Vision 특징과 언어(질문)를 연결하는 핵심 모듈. 입력으로 1024차원의 특징 벡터를 받도록 hidden_size=1024로 설정  
- **언어 생성 모듈 (Frozen)**  
  - **Projection Layer (llm_proj)**: Q-Former의 출력(1024차원)을 LLM의 입력 차원(2048차원)으로 변환하는 `nn.Linear(1024, 2048)` 레이어  
  - **LLM**: Llama-3.2-1B 모델을 bitsandbytes를 통해 4-bit 양자화하여 로드

#### 3. 학습 파이프라인
ImageBind와 MA-LMM의 라이브러리 의존성 충돌(PyTorch, timm 버전 등)을 해결하기 위해, 각자의 역할을 독립된 가상환경에서 수행하는 2단계 파이프라인 방식을 채택했습니다.

##### 1단계: 임베딩 사전 추출 (in imagebind_env)
ImageBind 전용 가상환경에서, VisDial v1.0 데이터셋의 모든 이미지와 각 대화 턴의 질문 텍스트에 대한 Vision/Text 임베딩을 미리 추출하여 디스크에 `.pt` 파일로 저장합니다.

##### 2단계: 모델 파인튜닝 (in malmm env)
MA-LMM 학습용 가상환경에서, 사전 추출된 임베딩을 입력으로 받아 오직 `imagebind_fc`와 `turn_pe` 레이어만 학습시킵니다.

#### 4. 설치 및 실행 가이드

##### 4.1 환경 설정

###### imagebind_env 생성 (임베딩 추출용)
```bash
conda env create -f environment_imagebind.yml
conda activate imagebind_env
```

###### malmm_env 생성 (모델 파인튜닝용)

```bash
conda env create -f environment_malmm.yml
conda activate malmm
```

참고: environment_*.yml 파일은 conda env export > [파일이름].yml 명령어로 생성할 수 있습니다.

##### 4.2 데이터 준비 및 실행

	•	데이터셋 다운로드
lavis/datasets/visdial/ 폴더에 VisDial v1.0 어노테이션과 MS COCO 2014, VisualDialog_val2018 이미지를 다운로드한 후 압축 해제
	•	임베딩 추출
```bash
conda activate imagebind_env
python extract_visdial_embeddings.py
```
	•	모델 파인튜닝
```bash
conda activate malmm
cd MA-LMM
python -m lavis.tasks.run –cfg-path lavis/configs/tasks/finetune_visdial.yaml
```
	•	성능 평가
```bash
python evaluate.py \
–checkpoint “lavis/output/finetune_visdial/finetune_visdial/checkpoint_latest.pth” \
–llm_model_path “llm/llama-3.2-1B/Llama-3.2-1B”
```
#### 5. 실험 결과

모델	BLEU-4 Score
파인튜닝 전 (FC Layer만 학습)	14.96
파인튜닝 후 (VisDial로 FC+PE 학습)	24.80

분석: VisDial 데이터셋과 순서 임베딩을 이용한 파인튜닝을 통해 BLEU-4 점수가 약 65.8% 향상되었습니다.

#### 6. 향후 과제 (Future Work)
	•	Retrieval 로직 구현: Recency, Frequency, Saliency 기반 스코어링 함수를 구현하여 대화 맥락에 맞는 과거 기억을 동적으로 검색하는 모듈 추가
	•	Self-Reflection 적용: 검색된 기억을 <reminder> 태그로 프롬프트에 주입하고, 모델이 연쇄적 사고(CoT)를 통해 더 논리적인 답변 생성
	•	종합 평가: VisDial뿐만 아니라 ScienceQA 등 다양한 벤치마크 데이터셋으로 정량·정성 평가 수행

