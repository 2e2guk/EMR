"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.runners.runner_base import RunnerBase  # 또는 blip2_runner 확인
from lavis.common.dist_utils import get_rank
import logging


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def run(self):
        # 1. 모델 로드
        model_cls = registry.get_model_class(self.cfg.model.arch)
        model = model_cls.from_config(self.cfg)

        # 2. 데이터셋 로드
        datasets = registry.get_datasets(self.cfg.datasets)

        # 3. job_id 생성 (보통 run name 씀)
        job_id = self.cfg.run_cfg.name

        # 4. Runner 실행
        runner = RunnerBase(
            task=self,
            model=model,
            datasets=datasets,
            job_id=job_id,
            cfg=self.cfg
        )
        runner.train()