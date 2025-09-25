# Cechy i dane

EnergiCast dostarcza gotowe generatory cech czasu, pogody i pozycji słonecznej.

## make_energy_features

Funkcja `energicast.features.energy.make_energy_features` tworzy ramkę cech na podstawie
`DatetimeIndex`:

- Cechy kalendarzowe (`hour`, `dow`, `month`, sygnały sinus/cosinus).
- Wskaźniki `is_weekend` i `is_holiday` (z możliwością rozszerzenia listy świąt).
- Ramp-rate dla pozycji słońca i prognoz pogodowych.
- Integracja z `solar_position_features` i `simple_weather_features`.

## Cechy solarne (`data/pv.py`)

`solar_position_features` oblicza zenit, elewację i azymut słońca:

- Domyślnie korzysta z `pvlib.get_solarposition` (jeśli dostępne), a w razie braku
  zależności przechodzi w tryb fallback z modelem analitycznym.
- Dodaje również `solar_cos_zenith` oraz `solar_day_fraction` przydatne przy generacji
  mocy PV.

## Cechy pogodowe (`data/weather.py`)

`simple_weather_features` generuje deterministyczne sygnały temperatury, wiatru i
napromienienia (GHI), a także laguje je o 1, 6 i 24 godziny.

## Dane PV i pogodowe w pipeline

Pipeline scala cechy z generatorów w `_make_feature_matrix`. Własne generatory można
przekazać przy inicjalizacji `ForecastPipeline(feature_generators=[...])`. Każdy generator
powinien zwracać `DataFrame` indeksowany tym samym `DatetimeIndex`.

## Moduł `data.pv`

Oprócz `solar_position_features` moduł zapewnia fallback bez zewnętrznych API. Dzięki temu
budowanie dokumentacji nie wymaga połączenia sieciowego ani obecności bibliotek systemowych.
