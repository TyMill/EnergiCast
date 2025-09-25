# Scenariusze pogody

Moduł `energicast.scenarios.weather_ensemble` udostępnia prosty generator scenariuszy
pogodowych.

## naive_weather_ensemble

- Przyjmuje bazową ramkę cech pogodowych (`X_base`).
- Tworzy `n` kopii z dodanym szumem Gaussowskim (`scale`).
- Zwraca listę ramek danych, które można wykorzystać do propagacji niepewności przez model.

## Roadmapa

Docelowo moduł będzie rozszerzony o integrację z prawdziwymi ansamblami NWP i narzędziami do
propagacji kwantyli. Obecny placeholder umożliwia szybkie prototypowanie.
