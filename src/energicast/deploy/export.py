"""Export trained pipelines with metadata for deployment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ..pipeline import ForecastPipeline


def export_pipeline(pipeline: ForecastPipeline, target: Path, fmt: str = "pickle") -> None:
    """Persist the pipeline artefacts together with metadata."""

    if not pipeline._artifacts:  # pylint: disable=protected-access
        raise ValueError("Pipeline must be fitted before export")
    target = Path(target)
    pipeline.save(target)
    meta_path = target / "metadata.json"
    try:
        version = metadata.version("energicast")
    except metadata.PackageNotFoundError:  # pragma: no cover - local editable installs
        version = "0.0.1"
    payload = {
        "package_version": version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model": pipeline.model_name,
        "quantiles": pipeline.config.quantiles,
        "horizon": pipeline.config.horizon,
        "frequency": pipeline.config.freq,
        "format": fmt,
        "series": sorted(pipeline._artifacts.keys()),  # pylint: disable=protected-access
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


__all__ = ["export_pipeline"]
