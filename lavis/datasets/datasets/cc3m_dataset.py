import os
import json
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate


class CC3MDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, ann_path=None, vis_root=None, **kwargs):
        """
        vis_processor: Lavis 이미지 전처리기
        text_processor: Lavis 텍스트 전처리기
        ann_path: JSON annotation 경로
        vis_root: 이미지 루트 디렉터리
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # JSON 파일 로딩
        with open(ann_path, "r") as f:
            self.annotations = json.load(f)

        # 이미지 열리지 않는 경우 필터링
        self.samples = []
        for ann in self.annotations:
            image_path = os.path.join(self.vis_root, ann["image"])
            try:
                with Image.open(image_path) as img:
                    img.verify()
                self.samples.append(ann)
            except (UnidentifiedImageError, OSError):
                continue

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #     ann = self.samples[idx]
    #     image_path = os.path.join(self.vis_root, ann["image"])
    #
    #     try:
    #         # PIL Image 로딩
    #         image = Image.open(image_path).convert("RGB")
    #     except Exception as e:
    #         print(f"Error loading image {image_path}: {e}")
    #         # 에러 발생 시, 임의의 검은 이미지와 더미 캡션을 반환하여 학습이 멈추지 않게 함
    #         image = Image.new('RGB', (224, 224))
    #         ann['caption'] = ""
    #
    #     # Lavis 이미지 전처리기 적용
    #     # vis_processor가 반드시 텐서를 반환해야 함
    #     #image = self.vis_processor(image)
    #     # 표준 torchvision transform을 직접 정의해서 사용합니다.
    #     # =============================================================
    #     image_transform = transforms.Compose([
    #         transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #     ])
    #     image = image_transform(image)
    #     # =============================================================
    #
    #     # 캡션 처리
    #     caption = ann["caption"]
    #     text_input = self.text_processor(caption)
    #
    #     return {
    #         "image": image,
    #         "text_input": text_input,
    #         "text_output": text_input,
    #     }

    # 임베딩 직접 받는 부분
    def __getitem__(self, idx):
        ann = self.samples[idx]
        image_filename = ann["image"]
        caption = ann["caption"]
        base_filename = os.path.splitext(image_filename)[0]

        # 생성한 임베딩 폴더 경로
        embedding_dir = "/home/leegw/EMR_2/cc3m_imagebind_embeddings_v2"
        vision_embedding_path = os.path.join(embedding_dir, base_filename + "_vision.pt")
        text_embedding_path = os.path.join(embedding_dir, base_filename + "_text.pt")

        try:
            # 이미지 파일을 여는 대신, torch.load로 임베딩을 바로 불러옵니다.
            vision_embedding = torch.load(vision_embedding_path)
            text_embedding = torch.load(text_embedding_path)
        except Exception as e:
            # print(f"Error loading embedding for {base_filename}: {e}") # 디버깅용
            # 에러 발생 시, 0으로 채워진 더미 텐서 반환
            vision_embedding = torch.zeros(1024)  # ImageBind Vision 출력 차원
            text_embedding = torch.zeros(1024)  # ImageBind Text 출력 차원

        # 모델의 forward 함수에 전달할 데이터 형식
        return {
            "image_embedding": vision_embedding,
            "text_embedding": text_embedding,
            "text_input": caption,
            "text_output": caption,
        }



    def collater(self, samples):
        return default_collate(samples)