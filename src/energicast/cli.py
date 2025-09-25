"""Command line interface for training, backtesting and exporting pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml
from rich.console import Console

from .backtest import run_backtest
from .pipeline import ForecastPipeline

console = Console()
app = typer.Typer(help="EnergiCast utilities")


def _load_config(config_path: Path) -> tuple[ForecastPipeline, pd.DataFrame]:
    config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if "training" not in payload:
        raise ValueError("Configuration must contain a 'training' section")
    from .config import TrainingConfig  # imported lazily to avoid circular imports

    training_cfg = TrainingConfig(**payload["training"])
    data_cfg = payload.get("data", {})
    data_path = Path(data_cfg.get("path", ""))
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    read_kwargs = data_cfg.get("read_csv", {})
    df = pd.read_csv(data_path, **read_kwargs)
    pipeline = ForecastPipeline(config=training_cfg)
    return pipeline, df


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to YAML configuration"),
    out: Path = typer.Option(..., help="Directory to store the trained model"),
) -> None:
    """Train a forecasting pipeline and persist artefacts."""

    pipeline, df = _load_config(config)
    console.print("[bold]Training pipeline...[/bold]")
    pipeline.fit(df)
    out.mkdir(parents=True, exist_ok=True)
    pipeline.save(out)
    console.print(f"Model saved to [green]{out}[/green]")


@app.command()
def backtest(config: Path = typer.Option(...), out: Path = typer.Option(...)) -> None:
    """Run rolling-origin backtest and store metrics."""

    pipeline, df = _load_config(config)
    console.print("[bold]Running backtest...[/bold]")
    result = run_backtest(pipeline, df, output_dir=out)
    console.print(f"Backtest metrics saved to [green]{out}[/green]")
    console.print(result.summary)


@app.command()
def export(
    model_dir: Path = typer.Option(..., help="Directory with a trained pipeline"),
    fmt: str = typer.Option("pickle", help="Export format"),
    out: Optional[Path] = typer.Option(None, help="Target directory for exported artefacts"),
) -> None:
    """Export a trained model with metadata."""

    from .deploy.export import export_pipeline

    pipeline = ForecastPipeline.load(model_dir)
    target = out or (model_dir / f"export_{fmt}")
    export_pipeline(pipeline, target, fmt=fmt)
    console.print(f"Exported artefacts to [green]{target}[/green]")


@app.command()
def report(backtest_dir: Path = typer.Option(..., help="Directory with backtest outputs")) -> None:
    """Display a textual summary of a backtest run."""

    metrics_path = backtest_dir / "metrics.csv"
    summary_path = backtest_dir / "summary.json"
    if not metrics_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Backtest directory must contain metrics.csv and summary.json")
    metrics = pd.read_csv(metrics_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    console.print("[bold]Backtest summary[/bold]")
    console.print(summary)
    console.print("[bold]Per-series metrics[/bold]")
    console.print(metrics.groupby("series_id").mean())


if __name__ == "__main__":
    app()
