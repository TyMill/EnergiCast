# Plan dokumentacji

- `mkdocs.yml` – konfiguracja nawigacji, motywu Material oraz pluginów mkdocstrings.
- `docs/index.md` – strona główna z omówieniem biblioteki i architekturą.
- `docs/getting-started.md` – instrukcje instalacji i minimalny przykład pipeline.
- `docs/guides/*.md` – przewodniki tematyczne (pipeline, modele, AutoML, cechy, imputacja,
  metryki, backtest, CLI, deploy, hierarchie, scenariusze, benchmarki).
- `docs/contributing.md` – zasady stylu, testów i budowy dokumentacji.
- `docs/reference/*.md` – źródła referencji API generowane przez mkdocstrings.
- `.github/workflows/docs.yml` – automatyczne budowanie i publikacja na GitHub Pages.
- `pyproject.toml` (sekcja `docs`) – dodatkowe zależności wymagane do budowy dokumentacji.
- `README.md` – sekcja z instrukcjami uruchomienia dokumentacji lokalnie.
- `PLAN_DOKS.md`, `KOMENDY.md`, `STATUS.md` – meta dokumenty opisujące wykonane kroki i status.
