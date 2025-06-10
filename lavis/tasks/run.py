from lavis.common.config import parse_cfg
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import get_rank
import logging
from omegaconf import OmegaConf
from lavis.common import schedulers

def main():
    cfg = parse_cfg()  # 수정됨

    if get_rank() == 0:
        #logging.info("Config:\n" + cfg.pretty_text)
        logging.info("Config:\n" + OmegaConf.to_yaml(cfg))

    #task = registry.get_task_class(cfg.run_cfg.task)(cfg)
    task_cls = registry.get_task_class(cfg.run_cfg.task)
    task = task_cls.from_config(cfg)
    task.run()


if __name__ == "__main__":
    main()