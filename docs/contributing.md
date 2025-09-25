# Wkład / Contributing

Dziękujemy za zainteresowanie rozwojem EnergiCast! Poniżej najważniejsze wskazówki dla
kontrybutorów.

## Styl kodu

- Używamy `black` (linia 100 znaków) oraz `ruff` (`E`, `F`).
- Docstringi piszemy w stylu Google – jest to wymagane przez `mkdocstrings`.
- Nowe moduły powinny eksportować główne symbole poprzez `__all__`.

## Testy

Przed zgłoszeniem PR uruchom:

```bash
pip install -e .[dev,docs]
pytest
mkdocs build --strict
```

## Dokumentacja

- Pliki narracyjne znajdują się w `docs/` (przewodniki w `docs/guides/`).
- Referencję API generujemy przy pomocy `mkdocstrings`; wystarczy dodać blok `:::` w pliku
  z katalogu `docs/reference/`.
- Jeśli dodajesz nowe moduły, pamiętaj o krótkich docstringach (klasy, metody, funkcje).

## Budowanie dokumentacji lokalnie

```bash
pip install -e .[docs]
mkdocs serve
```

`mkdocs serve` uruchomi serwer pod `http://127.0.0.1:8000/` z automatycznym przeładowaniem.

## Zgłaszanie błędów

Otwórz issue z minimalnym przykładem i logami. W przypadku błędów w dokumentacji dołącz
link do konkretnej strony oraz propozycję poprawki.
