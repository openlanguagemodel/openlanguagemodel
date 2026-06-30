"""Loss functions for OLM training."""

from olm.train.losses.base import LossBase
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.kllloss import KLLoss
from olm.train.losses.mce import MaskedCELoss
from olm.train.losses.zloss import ZLoss
from olm.train.losses.load_balance_loss import LoadBalanceLoss
from olm.train.losses.mtp_loss import MTPLoss

__all__ = [
    "LossBase",
    "CrossEntropyLoss",
    "KLLoss",
    "MaskedCELoss",
    "ZLoss",
    "LoadBalanceLoss",
    "MTPLoss",
]
