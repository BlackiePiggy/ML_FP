import math

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = self.get_lr(self.base_lrs[i])

    def get_lr(self, base_lr):
        if self.current_step < self.warmup_steps:
            return base_lr * float(self.current_step) / float(max(1, self.warmup_steps))
        progress = (self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
