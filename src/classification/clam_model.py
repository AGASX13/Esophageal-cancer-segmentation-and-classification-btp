from __future__ import annotations


def torch_required() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for classification stage. Install `torch` first.") from e


class CLAM_SB:
    """
    Minimal placeholder for CLAM-SB model.

    Your report uses CLAM (Lu et al.). Full implementation will include:
    - attention network
    - instance-level clustering loss
    - bag-level classifier head
    - heatmap export
    """

    def __init__(self, feature_dim: int, num_classes: int = 2):
        torch_required()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        raise NotImplementedError("CLAM-SB architecture will be implemented next.")

