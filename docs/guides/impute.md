# Imputacja

EnergiCast koncentruje się na uzupełnianiu luk w danych energetycznych, gdzie braki są częste
w logach licznikowych i SCADA.

## GapFiller (`energicast.impute.synth.GapFiller`)

- **Strategia**: forward fill i backward fill ograniczone parametrem `max_gap_hours`,
  a następnie uzupełnienie średnią dla danej godziny (`hod`).
- **Ograniczenia**: opcjonalne `enforce_non_negative` przycina wyniki poniżej zera.
- **Interfejs**: standardowe metody `fit`, `transform`, `fit_transform` zgodne z API
  scikit-learn.

## Praca z brakami

Pipeline automatycznie aplikuje `GapFiller` podczas treningu (`_fit_single_series`).
Walidacja krocząca odtwarza proces dla każdego okna, aby uniknąć przecieku informacji.

## Testy z „dziurami”

W repozytorium testy jednostkowe (w `tests/`) generują sztuczne braki i weryfikują, że
`GapFiller` zwraca serię pozbawioną `NaN`. Własne eksperymenty można przeprowadzić, tworząc
podzbiory danych z brakami i wykorzystując `fit_transform`.
