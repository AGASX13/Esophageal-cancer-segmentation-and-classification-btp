from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """
    Resolve the repository root as:
    <root>/src/common/paths.py -> <root>
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def config(self) -> Path:
        return self.root / "config"

    # Stage-specific conveniences
    @property
    def tcga_wsi_raw(self) -> Path:
        return self.data_raw / "tcga_esca_wsi"

    @property
    def tcga_annotations_raw(self) -> Path:
        return self.data_raw / "tcga_esca_annotations"


def get_paths() -> ProjectPaths:
    return ProjectPaths(root=project_root())

