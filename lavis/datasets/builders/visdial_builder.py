from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.visdial_dataset import VisDialDataset
from lavis.common.registry import registry

@registry.register_builder("visdial_malmm")
class VisDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialDataset

    # 평가(eval)에도 동일한 데이터셋 클래스를 사용합니다.
    eval_dataset_cls = VisDialDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/visdial/defaults.yaml",
    }