"""Loss functions for OLM training."""

from olm.train.losses.base import LossBase
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.kllloss import KLLoss
from olm.train.losses.mce import MaskedCELoss
from olm.train.losses.zloss import ZLoss
from olm.train.losses.load_balance import LoadBalanceLoss
from olm.train.losses.sequence_load_balance import SequenceLoadBalanceLoss
from olm.train.losses.mtp import MTPLoss

__all__ = [
    "LossBase",
    "CrossEntropyLoss",
    "KLLoss",
    "MaskedCELoss",
    "ZLoss",
    "LoadBalanceLoss",
    "SequenceLoadBalanceLoss",
    "MTPLoss",
]
