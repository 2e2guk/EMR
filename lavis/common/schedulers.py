from lavis.common.registry import registry
from torch.optim.lr_scheduler import LambdaLR
import math


@registry.register_lr_scheduler("cosine")
def cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, **kwargs):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)