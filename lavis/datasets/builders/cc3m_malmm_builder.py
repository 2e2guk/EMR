# lavis/datasets/builders/cc3m_malmm_builder.py

import os
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.datasets.datasets.cc3m_dataset import CC3MDataset
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, load_dataset_config

@registry.register_builder("cc3m_malmm")
class CC3MMALMMDatasetBuilder(BaseDatasetBuilder):
    # 데이터셋 구현체
    train_dataset_cls = CC3MDataset
    eval_dataset_cls  = CC3MDataset

    # 기본 설정 파일 경로 (repo 루트 기준)
    DATASET_CONFIG_DICT = {
        "default": "lavis/configs/datasets/cc3m/cc3m_malmm.yaml"
    }

    def __init__(self, override_cfg):
        """
        override_cfg: Task YAML에서 넘어온 부분 (경로만 지정된 build_info)
        """
        # 1) 기본 cc3m_malmm.yaml 로드
        default_cfg = load_dataset_config(self.DATASET_CONFIG_DICT["default"])
        # 2) override_cfg (OmegaConf) 와 머지
        merged = OmegaConf.merge(default_cfg, override_cfg)
        # 3) BaseDatasetBuilder 초기화 (여기서 self.config.data_type, processors, build_info 세팅)
        super().__init__(merged)

    def _download_ann(self):
        # 다운로드 스텝 건너뛰기
        return

    def _download_vis(self):
        # 다운로드 스텝 건너뛰기
        return

    def build(self):
        """
        BaseDatasetBuilder.build_datasets() 대신 직접 build() 구현.
        merged된 self.config.build_info를 사용.
        """
        # train split 정보
        ann_info = self.config.build_info.annotations.train
        img_info = self.config.build_info.images.train

        return {
            "train": self.train_dataset_cls(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                ann_path=ann_info.storage,
                vis_root=img_info.storage,
            )
        }