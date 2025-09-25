import pathlib
import shutil

from typer.testing import CliRunner

from energicast.cli import app

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cli_train_backtest_export_report(tmp_path):
    runner = CliRunner()
    config_src = ROOT / "examples" / "pv_config.yaml"
    data_src = ROOT / "examples" / "pv_sample.csv"
    config_path = tmp_path / "config.yaml"
    data_path = tmp_path / "pv_sample.csv"
    shutil.copy(config_src, config_path)
    shutil.copy(data_src, data_path)

    model_dir = tmp_path / "model"
    result = runner.invoke(app, ["train", "--config", str(config_path), "--out", str(model_dir)])
    assert result.exit_code == 0
    assert (model_dir / "config.json").exists()

    backtest_dir = tmp_path / "backtest"
    result = runner.invoke(
        app,
        ["backtest", "--config", str(config_path), "--out", str(backtest_dir)],
    )
    assert result.exit_code == 0
    assert (backtest_dir / "metrics.csv").exists()

    result = runner.invoke(
        app,
        ["export", "--model-dir", str(model_dir), "--fmt", "pickle"],
    )
    assert result.exit_code == 0
    export_dir = model_dir / "export_pickle"
    assert (export_dir / "metadata.json").exists()

    result = runner.invoke(app, ["report", "--backtest-dir", str(backtest_dir)])
    assert result.exit_code == 0
