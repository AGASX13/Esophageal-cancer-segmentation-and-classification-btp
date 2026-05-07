from __future__ import annotations


def torch_required() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for segmentation stage. Install `torch` first.") from e


def build_unet_resnet34(num_classes: int):
    """
    Placeholder builder for a U-Net with ResNet34 encoder.

    Recommended implementation path:
    - Use `segmentation-models-pytorch` (smp) with encoder_name='resnet34'
      OR implement your own U-Net + torchvision resnet encoder.
    """
    torch_required()
    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Install `segmentation-models-pytorch` for U-Net ResNet34 builder."
        ) from e

    # For PanNuke-like masks you may want activation=None and apply softmax in loss.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=int(num_classes),
    )
    return model

