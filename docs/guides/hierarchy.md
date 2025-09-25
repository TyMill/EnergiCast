# Hierarchie i rekonsyliacja

EnergiCast przewiduje obsługę hierarchicznych prognoz (np. licznik → stacja → system).

## Rekonsyliacja MinT

Moduł `energicast.hier.reconcile` zawiera funkcję `mint_reconcile(bottom_forecasts, S)`:

- `bottom_forecasts` – DataFrame z prognozami dla węzłów dolnych (kolumny) i znacznikiem
  czasu w indeksie.
- `S` – macierz sumująca (kształt: wszystkie węzły × węzły dolne).
- Wynik to prognozy dla wszystkich poziomów uzyskane przez mnożenie macierzy.

## Roadmapa

Aktualna implementacja jest placeholderem i nie zawiera estymacji kowariancji (pełne MinT).
Integracja z pipeline będzie obejmowała przekazywanie macierzy `S` oraz wag.
