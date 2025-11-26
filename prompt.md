# LocaLLM Bench – Notatka operacyjna

## Cel
Minimalny benchmark lokalnych modeli LLM przez Ollamę: porównanie jakości i czasu odpowiedzi na prostych zestawach zadań (instrukcje, kod z auto-testami, logika, PL), zapis surowych wyników oraz raport z accuracy i metrykami pomocniczymi.

## Stan projektu
- Runtime: Python 3.12.
- Połączenie: biblioteka `ollama` (`client.generate(stream=False)`), host z `config.yaml`.
- Pliki kluczowe: `runner.py`, `metrics.py`, `report.py`, `config.yaml`, `tests/*.json`.
- Start: `start.sh` (menu: install, benchmark, raport) lub bezpośrednio `.venv/bin/python runner.py`.
- Debug wspierany z configu (`debug`, `debug_models`, `debug_categories`, `debug_task_limit`); logi do stdout i `runner.log`.

## Workflow
1) Instalacja: `bash install.sh` (tworzy `.venv`, instaluje z `requirements.txt`).
2) Uruchomienie: `bash start.sh` → opcja 2 (benchmark) lub 3 (raport). Możesz też: `.venv/bin/python runner.py`.
3) Debug: ustaw `debug: true` w `config.yaml` i opcjonalnie `debug_models`/`debug_categories`/`debug_task_limit`; runner loguje odpowiedzi modeli dla szybkiego sprawdzenia.
4) Raport: po `runner.py` uruchom `report.py` – generuje `report.md` (tabelki Markdown), liczy accuracy, metrykę `contains_all`, statystyki auto-testów kodu, zapisuje wycięte fragmenty kodu do `artifacts/`.

## Konfiguracja (`config.yaml`)
- `ollama_host`: URL do instancji Ollama.
- `models`: lista modeli do pełnego biegu.
- `tests_dir`: katalog z zestawami JSON (`instruction.json`, `code.json`, `polish.json`, ...).
- Debug: `debug` (bool), `debug_models`, `debug_categories`, `debug_task_limit` (liczba zadań/kategorię).

## Testy wejściowe (`tests/*.json`)
Każdy wpis: `prompt` oraz klucze scoringu:
- `expected` (exact), `contains_all`, `contains_any`, `asserts` (alias na contains_all), `code_tests` (lista testów: `call`, `expected`; uruchamiane na wygenerowanym kodzie).

## Scoring i raport
- `metrics.py`: `score_task` łączy kryteria (exact/contains_all/contains_any/asserts) i umie odpalić proste `code_tests` na wyciętym kodzie (wyodrębnianie z bloków ```python).
- `report.py`: wczytuje `results.json`, liczy accuracy per model/kategoria oraz overall, podsumowuje `contains_all` i `code_tests`, zapisuje kod do `artifacts/*.py`, wypisuje błędy sieci/wykonania.

## Zasady komunikacji
- Pisz kod i dokumentację po angielsku; rozmowa po polsku.
