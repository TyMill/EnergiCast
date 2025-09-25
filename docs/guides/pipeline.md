# Pipeline

`energicast.pipeline.ForecastPipeline` to klej spinający wszystkie komponenty biblioteki.
Pipeline buduje kompletny cykl: ładuje dane, imputuje braki, generuje cechy, trenuje
model, wykonuje walidację kroczącą oraz zapisuje/odtwarza artefakty.

## Konfiguracja

- `TrainingConfig` definiuje częstotliwość (`freq`), horyzont (`horizon`), listę kwantyli,
  a także model (`model`) i jego parametry (`model_params`).
- `DatasetConfig` opisuje kolumny (`time_column`, `target_column`, `series_id_column`),
  znane kowariaty (`known_covariates`) oraz wymagania co do kompletności historii.
- `ValidationConfig` określa tryb walidacji; pipeline implementuje wariant
  `rolling` poprzez `_rolling_origin_windows`.

## Przepływ danych

1. **Ładowanie i normalizacja** – `_prepare_single_series` porządkuje indeks
   czasowy, wymusza strefę czasową oraz uzupełnia brakujące znaczniki czasu.
2. **Imputacja** – `GapFiller.fit_transform` z modułu `energicast.impute.synth`
   uzupełnia luki przy użyciu kombinacji forward/backward fill i średnich godzinowych.
3. **Cechy** – `_make_feature_matrix` wywołuje listę generatorów (domyślnie tylko
   `make_energy_features`) oraz łączy je z kowariatami znanymi w przyszłości.
4. **Trenowanie** – `_fit_single_series` instancjonuje model z rejestru (np.
   `XGBForecaster`) i trenuje go na imputowanym sygnale.
5. **Walidacja** – `_maybe_validate_series` tworzy okna rolling-origin i zapisuje
   prognozy w `validation_results_`, które później wykorzystuje `backtest.run_backtest`.
6. **Prognozowanie** – `predict` generuje macierz kwantyli dla każdego szeregu,
   korzystając z zachowanych artefaktów (`PipelineArtifacts`).
7. **Persistencja** – `save` zapisuje konfigurację i modele per-seria, natomiast
   `load` rekonstruktuje pipeline z katalogu.

## Praca z wieloma seriami

Jeżeli `series_id_column` jest ustawione, pipeline rozdziela ramkę danych na grupy i
utrzymuje osobne artefakty dla każdej serii. Walidacja i predykcje zwracają wynik w
postaci MultiIndexu (`series_id`, `timestamp`).

## Integracja z CLI i backtestem

Komendy `energicast.cli` tworzą pipeline na podstawie pliku YAML, uruchamiają trening
(`train`) lub backtest (`backtest`), a następnie zapisują wyniki do katalogów z artefaktami.
`backtest.run_backtest` oczekuje, że `validation_results_` zawiera rolling-origin windows,
co zapewnia konfiguracja walidacji ustawiona na `method="rolling"`.
