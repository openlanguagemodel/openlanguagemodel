from setuptools import find_packages


def test_train_losses_public_exports():
    from olm.train.losses import (
        CrossEntropyLoss,
        KLLoss,
        LossBase,
        MaskedCELoss,
        ZLoss,
    )

    assert LossBase is not None
    assert CrossEntropyLoss is not None
    assert KLLoss is not None
    assert MaskedCELoss is not None
    assert ZLoss is not None


def test_train_losses_is_in_discovered_packages():
    packages = find_packages(where="src", include=["olm*"])

    assert "olm.train.losses" in packages
