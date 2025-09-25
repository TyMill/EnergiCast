# Deploy/Export

Moduł `energicast.deploy.export` pozwala przygotować artefakty modelu do wdrożenia.

## export_pipeline

```python
from energicast.deploy.export import export_pipeline
from energicast.pipeline import ForecastPipeline

pipeline = ForecastPipeline.load("runs/model_xgb")
export_pipeline(pipeline, "runs/export_xgb", fmt="pickle")
```

Funkcja:

- Upewnia się, że pipeline został wcześniej dopasowany (`_artifacts`).
- Kopiuje artefakty modelu (`pipeline.save`).
- Dodaje plik `metadata.json` z wersją pakietu (`importlib.metadata.version`), listą serii,
  horyzontem i kwantylami.

## Wersjonowanie i metadane

- `package_version` pozwala śledzić, z jakiej wersji EnergiCast pochodzi model.
- `exported_at` przechowuje znacznik czasu (UTC).
- `format` można wykorzystać do różnicowania formatów eksportu (np. `pickle`, `onnx`).

## Integracja z CLI

Polecenie:

```bash
python -m energicast.cli export --model-dir runs/model_xgb --fmt pickle --out runs/export_xgb
```

Po eksporcie katalog zawiera strukturę identyczną jak przy `ForecastPipeline.save`
(`config.json`, `models/`) oraz plik `metadata.json`.
