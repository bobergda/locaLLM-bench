# LocaLLM Bench – Notatka operacyjna

## Cel
Minimalny benchmark lokalnych modeli LLM przez Ollamę: porównanie jakości i czasu odpowiedzi na prostych zestawach zadań (instrukcje, kod, logika, PL), zapis surowych wyników oraz raport z accuracy.

## Stan projektu
- Runtime: Python 3.12.
- Połączenie: biblioteka `ollama` (`client.generate(stream=False)`), host z `config.yaml`.
- Pliki kluczowe: `runner.py`, `metrics.py`, `report.py`, `config.yaml`, `tests/*.json`.
- Debug wspierany z configu (`debug`, `debug_models`, `debug_categories`, `debug_task_limit`); logi do stdout i `runner.log`.

## Workflow
1) Instalacja: `bash install.sh` (tworzy `.venv`, instaluje z `requirements.txt`).
2) Uruchomienie: `bash start.sh` → opcja 2 (benchmark) lub 3 (raport). Możesz też: `.venv/bin/python runner.py`.
3) Debug: ustaw `debug: true` w `config.yaml` i opcjonalnie `debug_models`/`debug_categories`/`debug_task_limit`; runner loguje odpowiedź modelu dla szybkiego sprawdzenia.
4) Raport: po `runner.py` uruchom `report.py` – generuje `report.md` (tabelka Markdown) i `report.csv` z accuracy.

## Konfiguracja (`config.yaml`)
- `ollama_host`: URL do instancji Ollama.
- `models`: lista modeli do pełnego biegu.
- `tests_dir`: katalog z zestawami JSON (`instruction.json`, `code.json`, `polish.json`, ...).
- Debug: `debug` (bool), `debug_models`, `debug_categories`, `debug_task_limit` (liczba zadań/kategorię).

## Testy wejściowe (`tests/*.json`)
Każdy wpis: `prompt` oraz jeden z kluczy scoringu:
- `expected` (exact), `contains_all`, `contains_any`.

## Scoring i raport
- `metrics.py`: `score_task` (exact/contains_all/contains_any) + `compute_accuracy` per model/kategoria.
- `report.py`: wczytuje `results.json`, liczy accuracy i zapisuje CSV/Markdown.

## Zasady komunikacji
- Pisz kod i dokumentację po angielsku; rozmowa po polsku.
- Przy zmianach istniejących plików pokazuj tylko nowe/zmienione fragmenty, bez przepisywania całych plików jeśli nie trzeba.
- Przy nowych plikach dawaj pełną zawartość z informacją „Zapisz to jako: NAZWA_PLIKU”.
- Unikaj gadania – najpierw konkretny kod/treść, potem krótki komentarz, wskazuj pliki.
