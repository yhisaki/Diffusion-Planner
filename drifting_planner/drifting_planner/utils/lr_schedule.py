from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, MultiplicativeLR, SequentialLR


def CosineAnnealingWarmUpRestarts(
    optimizer: Optimizer,
    epoch: int,
    warm_up_epoch: int,
    start_factor: float = 0.1,
) -> SequentialLR:
    assert epoch >= warm_up_epoch
    T_warmup = warm_up_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, fixed_scheduler], milestones=[T_warmup]
    )

    return scheduler
