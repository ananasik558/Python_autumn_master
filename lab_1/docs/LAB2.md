# ЛР2: анализ сходства и воспроизводимое сравнение методов

Документ фиксирует реализацию, протокол эксперимента, результаты и ограничения.
Версия отчёта: **2.2.0**. Корпус учебный и синтетический: метрики ниже нельзя
переносить на реальных студентов или использовать как вердикт о нарушении.

## 1. Что добавлено к ЛР1

| Компонент | Файл | Назначение |
|---|---|---|
| Токенизация кода | `src/submission_system/tokenizer.py` | `tokenize`, отбрасывание комментариев и служебных токенов, отделение строк шаблона, n-граммы |
| Модели корпуса и JSONL | `src/submission_system/similarity/corpus.py` | Потоковое чтение, `CorpusSubmission`, `CorpusPair`, сводка корпуса |
| Token baseline | `src/submission_system/similarity/baseline.py` | Jaccard по 3-граммам, evidence с диапазонами строк, сведения о шаблоне |
| TF-IDF | `src/submission_system/similarity/tfidf.py` | Fit только на tuning, cosine, OOV-предупреждения |
| Пары | `src/submission_system/similarity/pairs.py` | Только разные студенты, одно задание, повторные версии отдельно |
| Признаки пары | `src/submission_system/similarity/features.py` | Числовые признаки для ML, метка происхождения отдельно |
| Retrieval для LLM | `src/submission_system/similarity/retrieval.py` | TF-IDF-кандидаты до генерации feedback, детерминированная заглушка |
| Оценка | `src/submission_system/similarity/evaluation.py` | Precision/recall/F1, ROC-AUC, `review_required`, матрица ошибок, доля неопределённых, порог только на tuning |
| Разбор ошибок | `src/submission_system/similarity/error_analysis.py` | FP/FN с объяснениями, обязательный случай общего шаблона |
| ML-классификатор | `src/submission_system/similarity/ml.py` | Логистическая регрессия на признаках пары, bootstrap и grouped-CV |
| Логирование | `src/submission_system/similarity/mlflow_logging.py` | MLflow при наличии пакета, JSON-fallback иначе |
| Эксперимент | `src/submission_system/similarity/experiment.py` | Полный воспроизводимый прогон и сохранение отчётов |
| CLI | `src/submission_system/similarity/cli.py` | `evaluate`, `retrieve`, `errors`, `ml` |
| Корпус | `data/corpus/*.jsonl` | 4 задания, 60 сдач, 40 пар |
| Сборка корпуса | `scripts/build_corpus.py` | Детерминированная генерация корпуса |
| Замер памяти | `scripts/benchmark_memory.py` | Потоковое чтение против полного индекса |
| Профилирование | `scripts/profile_pipeline.py` | Время и пиковая память на 100/1000/5000 записях |

## 2. Учебный корпус

Корпус лежит в `data/corpus/`:

* `assignments.jsonl` — 4 задания, у `hw_template` есть разрешённый шаблон;
* `submissions.jsonl` — 60 сдач, отсортированы по `assignment_id`;
* `pairs.jsonl` — 40 пар с происхождением варианта.

Характеристики:

| Показатель | Значение |
|---|---:|
| Пар | 40 |
| Сдач | 60 |
| Семейств вариантов | 20 |
| Пар с разрешённым шаблоном | 10 |
| Tuning-пар / evaluation-пар | 18 / 22 |
| Tuning-семейств / evaluation-семейств | 9 / 11 |

Распределение происхождения:

| `derivation_relation` | Пар |
|---|---:|
| `exact_copy` | 6 |
| `formatting` | 7 |
| `renaming` | 7 |
| `template_only` | 10 |
| `independent` | 10 |

У каждой пары сохранены `assignment_id`, `family_id`, `split`,
`derivation_relation`, `template_only`. Сплит задан на уровне `family_id`,
поэтому производные одной семьи не попадают одновременно в tuning и evaluation.
`derivation_relation` — это происхождение примера, а не вердикт «плагиат».
Для каждой пары в `pair_comparisons.jsonl` сохраняются совпавшие n-граммы,
диапазоны строк слева/справа и сведения о шаблоне.

## 3. Потоковое чтение и память

`iter_jsonl` читает по одной строке. `iter_candidate_pairs_streaming`
использует `itertools.groupby` и удерживает только текущее задание; для этого
файл должен быть отсортирован по `assignment_id`. Полный индекс
`submission_id -> submission` строится только там, где он нужен (например, для
доступа к паре по идентификатору). Это не делает память константной: если все
сдачи лежат в одном задании или строится полный индекс, потребление растёт
линейно.

Замер `scripts/benchmark_memory.py` (tracemalloc, КиБ):

| Записей | Streaming | Полная загрузка | Полный индекс | Streaming-пары |
|---:|---:|---:|---:|---:|
| 60 | 53.7 | 108.7 | 109.8 | 75.8 |
| 300 | 48.4 | 456.4 | 460.5 | 177.4 |
| 600 | 49.0 | 884.0 | 891.6 | 322.9 |

Streaming-чтение остаётся примерно постоянным, полная загрузка и индекс растут
линейно. Streaming-пары растут вместе с размером окна одного задания. Значения
включают накладные расходы tracemalloc и приведены для сравнения порядков, а не
как точный бюджет памяти.

## 4. Методы

### 4.1 Token baseline

1. Код разбирается стандартным `tokenize`.
2. Отбрасываются `COMMENT`, `NL`, `NEWLINE`, `INDENT`, `DEDENT`, `ENCODING`,
   `ENDMARKER`.
3. Строки, совпавшие с `allowed_template`, выносятся в отдельный список и не
   участвуют в n-граммах.
4. Строятся 3-граммы значимых токенов.
5. Сходство: `|A ∩ B| / |A ∪ B|`. Если объединение пусто — `score=None`,
   а не `0`: это отсутствие данных.
6. Evidence содержит n-грамму и номера строк в обеих работах.

### 4.2 TF-IDF + cosine

* Словарь и IDF обучаются только на `split == "tuning"`.
* Evaluation-сдачи проходят только `transform`; повторный `fit` не вызывается.
* `max_df=0.9` фиксирован до эксперимента и отбрасывает почти служебную
  лексику (`def`, `return`, `for`), которая встречается почти везде.
* Cosine считается по разреженным векторам; пустой вектор даёт `None`.
* Если доля OOV-токенов больше 50%, в результат добавляется предупреждение:
  словарь обучен на другом наборе.
* Evidence — общие термы с номерами строк.

### 4.3 Признаки пары для ML

`pair_features.jsonl` содержит `token_jaccard`, `tfidf_cosine`,
`method_disagreement`, число общих/объединённых n-грамм, длины и их отношения,
число строк шаблона. `reference_label` хранится отдельно и не входит в
`features`, чтобы не подменять признаки меткой.

### 4.4 Retrieval для LLM

`TfidfIndex.retrieve` находит кандидатов по cosine, исключая self-match и
работы других студентов. Только после этого `RetrievalFeedbackGenerator`
формирует текст с кандидатами и evidence. Реальный вызов LLM не выполняется:
зафиксирован интерфейс и порядок шагов. Ответ содержит оговорку, что это
обнаруженное сходство, а не вердикт о нарушении.

## 5. Протокол эксперимента

1. TF-IDF обучается на tuning-сдачах.
2. Для каждой пары считаются baseline и TF-IDF.
3. На tuning выбирается порог каждого метода по F1 (пороги — середины между
   соседними score, вариант «все positive» исключён).
4. Тот же порог применяется к evaluation без изменений.
5. Считаются precision, recall, F1, accuracy и ROC-AUC.
6. Метод выбирается по tuning F1; evaluation используется только для отчёта.

## 6. Результаты

Команда:

```bash
python -m submission_system.similarity.cli evaluate --corpus data/corpus
```

| Метод | Порог (tuning) | Tuning F1 | Eval precision | Eval recall | Eval F1 | Eval AUC |
|---|---:|---:|---:|---:|---:|---:|
| `token_3gram_jaccard` | 0.086 | 0.867 | 0.762 | 1.000 | 0.865 | 0.927 |
| `tfidf_cosine` | 0.223 | 0.903 | 0.727 | 1.000 | 0.842 | 0.875 |

По tuning F1 выбран TF-IDF, но на evaluation лучше baseline по F1 и AUC.
Это ожидаемая демонстрация того, почему нельзя выбирать метод и порог на
evaluation и почему нужен отложенный набор.

Средний score по происхождению (evaluation):

| Происхождение | Baseline | TF-IDF |
|---|---:|---:|
| `exact_copy` | 1.000 | 1.000 |
| `formatting` | 0.930 | 0.954 |
| `renaming` | 0.197 | 0.482 |
| `template_only` | 0.297 | 0.699 |
| `independent` | 0.145 | 0.450 |

Интерпретация:

* Оба метода уверенно ловят точную копию и изменение форматирования.
* Baseline плохо переносит переименование: 3-граммы меняются вместе с именами.
* TF-IDF лучше на переименовании, но даёт высокий cosine независимым решениям
  одной задачи из-за общей лексики и структуры. Это видно по `independent`.
* После корректного исключения строк шаблона baseline перестаёт «награждать»
  пары за общий шаблон; TF-IDF всё ещё даёт им повышенный score, потому что
  собственная логика использует сходный набор токенов.
* Низкая precision при полном recall означает много ложных срабатываний.
  Поэтому score — сигнал для проверки, а не решение о нарушении.

## 7. Приёмка этапа А

Проверки вынесены в `tests/test_similarity_acceptance.py`:

| Пункт плана | Проверка | Результат |
|---|---|---|
| Точная копия получает максимальное непустое сходство | Все 6 пар `exact_copy`: baseline и TF-IDF дают `score=1.0`, evidence непустой | PASS |
| Общий шаблон не превращается в необъяснённое обвинение | У всех 10 `template_only`-пар есть `template_evidence`, `template_note`; строки шаблона исключены из признаков | PASS |
| Пустые признаки и некорректный код обработаны | Пустой/только-комментарии код даёт `score=None` и warning; битый код не роняет анализ | PASS |
| Top-k не содержит саму работу и сдачи другого задания | `TfidfIndex.retrieve` исключает query, same-student и другой `assignment_id` | PASS |
| Evidence указывает на существующие строки исходников | Для всех 40 пар проверены диапазоны строк evidence и `template_evidence` | PASS |

Дополнительно `template_note` и предупреждение TF-IDF явно объясняют, что
совпадение разрешённого шаблона не является основанием для обвинения.

## 8. Этап Б: review_required, ошибки и ML

### 8.1. Решение о ручном разборе

`review_required(score, threshold)` возвращает `True`, если `score >= threshold`,
`False`, если ниже, и `None`, если score неопределён (пустой код, нет общих
токенов). `None` не подменяется на `False`: отсутствие данных и отсутствие
сходства — разные ситуации.

Порог по-прежнему выбирается только на tuning по F1. Дополнительно в
`summary.json` сохраняются матрица ошибок, `review_rate` (доля пар, отправленных
на ручной разбор) и доля неопределённых результатов.

Результаты на evaluation:

| Метод | Порог | TP | FP | TN | FN | Precision | Recall | F1 | AUC | Review rate | Undefined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `token_3gram_jaccard` | 0.086 | 16 | 5 | 1 | 0 | 0.762 | 1.000 | 0.865 | 0.927 | 0.955 | 0.0 |
| `tfidf_cosine` | 0.223 | 16 | 6 | 0 | 0 | 0.727 | 1.000 | 0.842 | 0.875 | 1.000 | 0.0 |
| `logistic_regression_pair_features` | 0.513 | 13 | 2 | 4 | 3 | 0.867 | 0.812 | 0.839 | 0.906 | 0.682 | 0.0 |

`Review rate` показывает цену правила: baseline отправляет на проверку 95.5%
пар, TF-IDF — 100%, ML — 68.2%. При полном recall это значит, что почти все
пары попадают к преподавателю; такой режим допустим только как «показать
кандидатов», а не как автоматическое решение.

### 8.2. Разбор ошибок

Отчёт `artifacts/lab2/error_analysis.json` содержит все FP/FN, агрегаты и
6 репрезентативных случаев. Всего на выбранных порогах: 18 ошибок, из них
17 false positive и 1 false negative. Ошибки есть у обоих методов; в выборку
обязательно включён случай с разрешённым шаблоном.

| Метод | Пара | Split | Происхождение | Ошибка | Score | Объяснение |
|---|---|---|---:|---|---:|---|
| `token_3gram_jaccard` | F17-p2 | tuning | `template_only` | FN | 0.081 | Строки шаблона исключены, собственной логики мало; пара не попала в проверку |
| `tfidf_cosine` | F14-p1 | evaluation | `independent` | FP | 0.536 | Общая лексика и структура задачи подняли cosine |
| `token_3gram_jaccard` | F08-p2 | evaluation | `independent` | FP | 0.205 | Совпали служебные n-граммы обработки строк |
| `tfidf_cosine` | F10-p1 | evaluation | `independent` | FP | 0.483 | Общие имена стандартных функций и похожая структура решения |
| `tfidf_cosine` | F04-p2 | evaluation | `independent` | FP | 0.426 | Общая лексика задачи и совпавшие служебные n-граммы |
| `tfidf_cosine` | F14-p2 | evaluation | `independent` | FP | 0.423 | Ещё одна независимая пара того же семейства с общей лексикой |

Случай с шаблоном разобран отдельно. В разметке `template_only` относится к
`derived`, потому что обе работы используют общую заготовку. Формально это
false negative. Содержательно общий разрешённый шаблон не должен быть
обвинением: у пары есть `template_evidence`, `template_note` и предупреждение
TF-IDF. Именно поэтому в отчёте одновременно видны и метрика, и объяснение:
исключение шаблона из признаков не означает, что шаблон надо скрывать.

Главный вывод: на выбранных порогах почти все ошибки — ложные срабатывания на
независимых решениях одной задачи. Это ожидаемо для лексических методов и
означает, что score должен вести к ручной проверке, а не к вердикту.

### 8.3. Профилирование обработки

Скрипт `scripts/profile_pipeline.py` измеряет потоковый пайплайн: чтение JSONL,
токенизация, построение 3-грамм, TF-IDF-преобразование и cosine с эталоном.
Синтетические записи нужны только для нагрузки; качество на них не считается.

Параметры: размеры 100/1000/5000, 3 повтора на размер, `ngram_size=3`,
`max_features=5000`, `max_df=0.9`, `seed=0`, словарь обучен на 20 записях до
замера. Время и память измеряются отдельными сериями, чтобы `tracemalloc` не
искажал wall time.

| Записей | Wall time, median | Wall time, mean | Peak memory, median | Throughput, median |
|---:|---:|---:|---:|---:|
| 100 | 0.0179 s | 0.0179 s | 63.94 KiB | 5596 rec/s |
| 1000 | 0.1754 s | 0.1743 s | 63.91 KiB | 5701 rec/s |
| 5000 | 0.8484 s | 0.8504 s | 63.92 KiB | 5893 rec/s |

Время растёт линейно, пиковая память остаётся практически постоянной: пайплайн
не накапливает записи в списке. Это не отменяет ограничения полного индекса из
раздела 3: если строить `submission_id -> submission` для всех записей, память
будет расти линейно.

### 8.4. ML-классификатор против threshold baseline

`src/submission_system/similarity/ml.py` обучает логистическую регрессию на
8 признаках пары: `token_jaccard`, `tfidf_cosine`, `method_disagreement`,
`ngram_overlap_fraction_of_min`, `token_count_ratio`, `char_count_ratio`,
`template_matched_lines`, `missing_score_count`. Метка `reference_label` в
признаки не входит; число совпавших строк шаблона вычисляется по коду и
`allowed_template`, а не берётся из разметки пары. Пропуски заменяются
медианой tuning-набора, признаки стандартизуются.

Признак `template_matched_lines` может работать как shortcut: в корпусе все
`template_only`-пары имеют общий разрешённый шаблон. Поэтому grouped-CV по
семействам обязателен; без него оценка ML была бы оптимистичной.

Параметры: `learning_rate=0.2`, `epochs=600`, `l2=0.05`, порог выбирается по F1
только на tuning. Обучение — полный батч, без внешних библиотек. Результат
модели проверяется: NaN, бесконечность и значения вне `[0, 1]` отклоняются с
`ValueError`, а не подменяются нулём.

Сравнение на evaluation (тот же отложенный набор, что и у threshold-методов):

| Метод | Threshold | Precision | Recall | F1 | AUC | Review rate |
|---|---:|---:|---:|---:|---:|---:|
| `token_3gram_jaccard` | 0.086 | 0.762 | 1.000 | 0.865 | 0.927 | 0.955 |
| `tfidf_cosine` | 0.223 | 0.727 | 1.000 | 0.842 | 0.875 | 1.000 |
| `logistic_regression_pair_features` | 0.513 | 0.867 | 0.812 | 0.839 | 0.906 | 0.682 |

ML даёт более высокую precision и заметно меньшую долю пар на ручной разбор, но
проигрывает baseline по recall и немного по F1. Это не «победа ML»: на 22
evaluation-парах разница в 1–3 пары меняет метрику на несколько процентных
пунктов.

Нестабильность измерена двумя способами:

* bootstrap tuning-набора, 20 seed: F1 = 0.864 ± 0.047 (min 0.788, max 0.970);
* leave-one-family-out CV на tuning, 9 фолдов: F1 = 0.630 ± 0.457
  (min 0.000, max 1.000).

Большой разброс grouped-CV показывает, что модель чувствительна к составу
семейств. На таком объёме данных нельзя утверждать, что ML-подход устойчивее
threshold-правила. Корректный вывод: оба варианта — экспериментальные
кандидаты, а не готовая система принятия решений.

### 8.5. MLflow

`mlflow_logging.py` пишет run в локальный tracking backend MLflow: по умолчанию
это sqlite `mlflow.db`, а артефакты — в `mlruns/<experiment_id>/<run_id>/artifacts`.
В run сохраняются:

* **params** — `method_id`, список признаков, размеры tuning/evaluation,
  `split.split_unit`, `split.families_disjoint`, семейства split, гиперпараметры
  и выбранный на tuning порог;
* **metrics** — precision, recall, F1, accuracy, review rate, матрица ошибок и
  AUC для tuning/evaluation, а также bootstrap- и grouped-CV-сводки;
* **tags** — этап, задача `review_required`, метод и единица разбиения;
* **artifacts** — `split/ml_split.json`, `model/ml_results.json`,
  `model/ml_predictions.jsonl`, `errors/error_analysis.json`.

В выполненном run сохранено 22 параметра, 49 метрик и 4 артефакта; `run_id`,
`experiment_id` и `artifact_uri` записаны в `artifacts/lab2/mlflow_run.json`.
Среди параметров есть `report.version`, `report.config_hash`,
`report.corpus_fingerprint` и `report.template_fingerprint`.

После запуска создаётся `artifacts/lab2/mlflow_run.json` с `run_id`,
`experiment_id`, `tracking_uri`, `artifact_uri`, params, metrics и списком
залогированных артефактов. Это локальная опись run, а не замена MLflow.

Проверить run можно так:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Если MLflow не установлен, `mlflow_logging.py` по-прежнему создаёт
`mlflow_fallback.json` с теми же params/metrics/artifacts и явно указывает, что
реальный run не выполнялся. В текущем окружении MLflow установлен, поэтому
использован настоящий backend.

### 8.6. Проверки этапа Б

| Пункт | Где проверено | Результат |
|---|---|---|
| `review_required` и `None` для неопределённого score | `tests/test_similarity_stage_b.py` | PASS |
| Матрица ошибок, review rate и доля неопределённых | `summary.json`, тест | PASS |
| Не менее 3 ошибок, включая общий шаблон | `error_analysis.json`, тест | PASS |
| Профилирование 100/1000/5000, 3 повтора, median/mean | `profiling.json` | PASS |
| ML-классификатор против threshold baseline | `ml_results.json`, тест | PASS |
| Метка не входит в признаки ML | `ml_results.json` (`features`), тест | PASS |
| Параметры, метрики и split в MLflow | `mlflow_run.json`, `mlflow.db`, `mlruns/` | PASS |

### 8.7. Обязательные проверки

| Проверка | Где проверено | Результат |
|---|---|---|
| Одинаковый код даёт максимум | `test_required_same_code_gets_maximum_score` | PASS |
| Переименованные переменные не равны точной копии, но обнаруживаются | `test_required_renamed_variables_are_detected_but_not_exact` | PASS |
| Независимые решения ниже derived в среднем | `test_required_independent_solutions_are_lower_than_derived` | PASS |
| Разрешённый шаблон объясняется, а не скрывается | `test_required_allowed_template_is_explained`, `test_common_template_is_explained_not_hidden` | PASS |
| Пустой ввод остаётся неопределённым (`None`) | `test_required_empty_input_stays_undefined` | PASS |
| `SyntaxError` не роняет пайплайн | `test_required_syntax_error_does_not_crash_pipeline`, `test_tokenize_survives_broken_syntax` | PASS |
| self-match исключён в парах и retrieval | `test_required_self_match_is_excluded`, `test_top_k_excludes_self_other_assignment_and_same_student` | PASS |
| Повторная версия студента не сравнивается сама с собой | `test_required_repeated_student_versions_are_skipped`, `test_pair_generation_skips_self_match_and_counts_repeats` | PASS |
| Нет утечки семейств между tuning и evaluation | `test_required_no_family_leakage_between_tuning_and_evaluation`, `test_families_do_not_cross_tuning_evaluation_boundary` | PASS |
| Невалидный результат модели отклоняется | `test_required_invalid_model_scores_are_rejected`, `test_required_analysis_service_marks_invalid_model_result_failed` | PASS |
| MLflow payload содержит params, metrics и split | `test_required_mlflow_payload_contains_params_metrics_and_split` | PASS |
| Версия отчёта и отпечатки конфигурации | `test_required_report_has_version_and_config_fingerprint` | PASS |
| Новый шаблон меняет config, но не корпус | `test_required_template_change_updates_config_without_changing_corpus` | PASS |
| Сохранённый эксперимент воспроизводим | `test_required_saved_experiment_is_reproducible`, `test_required_ml_experiment_is_reproducible` | PASS |

Всего тестов: 77 (18 тестов ЛР1 + 59 тестов ЛР2); MLflow-интеграция
пропускается, если optional-зависимость не установлена.

## 9. Команды

```bash
# Все тесты
pytest tests/ -v

# Пересобрать корпус
python scripts/build_corpus.py

# Эксперимент и отчёты в artifacts/lab2/
python -m submission_system.similarity.cli evaluate --corpus data/corpus

# Retrieval + feedback для выбранной сдачи
python -m submission_system.similarity.cli retrieve \
    --corpus data/corpus --submission sub-F18-a --top-k 5

# Разбор ошибок (FP/FN + общий шаблон)
python -m submission_system.similarity.cli errors --corpus data/corpus

# Демонстрация нового разрешённого шаблона
python -m submission_system.similarity.cli template-demo --corpus data/corpus

# ML-классификатор, сравнение с порогами и MLflow-run
python -m submission_system.similarity.cli ml --corpus data/corpus --seeds 20

# Посмотреть MLflow-run
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Профилирование 100/1000/5000 записей, 3 повтора
python scripts/profile_pipeline.py --sizes 100,1000,5000 --repeats 3 --seed 0

# Память
python scripts/benchmark_memory.py --corpus data/corpus
```

Артефакты:

* `artifacts/lab2/summary.json` — метрики, пороги, матрицы ошибок, review rate;
* `artifacts/lab2/pair_comparisons.jsonl` — все пары с evidence;
* `artifacts/lab2/pair_features.jsonl` — признаки пар для ML;
* `artifacts/lab2/threshold_predictions.jsonl` — решение `review_required` по порогу;
* `artifacts/lab2/error_analysis.json` — FP/FN и разбор шаблонного случая;
* `artifacts/lab2/template_change_demo.json` — влияние нового шаблона на score и решения;
* `artifacts/lab2/profiling.json` — время и пиковая память на 100/1000/5000;
* `artifacts/lab2/ml_results.json` — модель, метрики, bootstrap и CV;
* `artifacts/lab2/ml_predictions.jsonl` — предсказания ML-классификатора;
* `artifacts/lab2/ml_split.json` — состав tuning/evaluation и признаки;
* `artifacts/lab2/mlflow_run.json` — run_id, tracking URI, params/metrics и список артефактов;
* `mlflow.db` и `mlruns/` — tracking backend и артефакты выполненного run;
* `artifacts/lab2/mlflow_fallback.json` — создаётся только без MLflow;
* `artifacts/lab2/memory.json` — замеры памяти.

## 10. Ограничения

1. Корпус синтетический и учебный; точность на реальных студентах не заявляется.
2. 40 пар — мало для статистически устойчивых выводов; метрики чувствительны
   к составу пар и выбранному порогу.
3. `derivation_relation` — происхождение примера, а не юридический/учебный
   вердикт. Метод не определяет «списал ли студент».
4. Шаблон сопоставляется по точному тексту строки; нечёткое форматирование
   шаблона не поддерживается.
5. TF-IDF обучается на ограниченном tuning-словаре; evaluation-код может
   содержать OOV-токены, что снижает score.
6. Реальный LLM не вызывается: реализован retrieval и детерминированная
   заглушка feedback.
7. Поддерживается Python-код; для других языков нужен отдельный tokenizer.
8. В разметке `template_only` отнесён к `derived`, потому что работы используют
   общую заготовку. Операционно общий разрешённый шаблон не должен быть
   обвинением; это объясняется через `template_note` и требует отдельного
   правила в реальной системе.
9. ML-набор очень мал: 18 tuning / 22 evaluation пары, 9 / 11 семейств.
   Bootstrap и grouped-CV показывают большой разброс, поэтому ML-результат —
   эксперимент, а не подтверждение готовности.
10. MLflow-run локальный (sqlite `mlflow.db`, артефакты в `mlruns/`):
    параметры, метрики и артефакты разбиения сохранены, но для командной работы
    нужен общий tracking server. JSON-fallback остаётся только для окружений
    без пакета.
11. Профилирование использует синтетические записи только для нагрузки;
    его время и память нельзя превращать в оценку качества методов.

## 11. Итоговый результат и защита

### 11.1. Что сдано

| Компонент | Результат |
|---|---|
| Анализатор сходства | Token 3-gram Jaccard и TF-IDF + cosine, evidence и `review_required` |
| Корпус | 40 пар, 60 сдач, 20 семейств; у каждой пары есть `derivation_relation` и `family_id` |
| Два метода | Сравнение на одном фиксированном наборе, порог выбирается только на tuning |
| Качество | Матрица ошибок, precision, recall, F1, AUC, review rate и доля `None` |
| Производительность | 100/1000/5000 записей, 3 повтора, median/mean, peak memory |
| ML | Логистическая регрессия на признаках пары, bootstrap и grouped-CV |
| Tracking | Реальный MLflow-run: params, metrics и split-артефакт |

### 11.2. Добавление нового шаблона на защите

Демонстрация не меняет файлы корпуса. Она строит второй прогон с изменённой
копией `Assignment`:

```bash
python -m submission_system.similarity.cli template-demo \
    --corpus data/corpus --output artifacts/lab2
```

Результат `artifacts/lab2/template_change_demo.json`:

| Показатель | До | После |
|---|---:|---:|
| `report_version` | 2.2.0 | 2.2.0 |
| `config_hash` | `95621d61c2908e61` | `2bd74b7e61aad0b0` |
| `template_fingerprint` | `d615f3d4a867013f` | `cdb7eca914cc4d74` |
| `corpus_fingerprint` | `2c24573c986d0592` | `2c24573c986d0592` |
| Baseline, средний score шаблонных пар | 0.258 | 0.056 |
| TF-IDF, средний score шаблонных пар | 0.644 | 0.431 |
| Baseline, `review_required` среди шаблонных пар | 9 | 7 |
| TF-IDF, `review_required` среди шаблонных пар | 10 | 7 |
| Evaluation F1 baseline / TF-IDF | 0.865 / 0.842 | 0.811 / 0.833 |

Изменились 7 решений `review_required`. Одна пара после исключения нового блока
стала неопределённой (`score=None`), а не «нулевой»: это разные состояния.
Порог после изменения конфигурации повторно выбирается на tuning, поэтому
меняются и решения, и метрики.

Вывод для защиты: шаблон — это конфигурация, а не подгонка ответа. Меняются
`config_hash` и `template_fingerprint`, а `corpus_fingerprint` остаётся прежним.
В реальном корпусе шаг тот же: обновить `allowed_template` в
`data/corpus/assignments.jsonl`, перезапустить `evaluate` и сравнить версию
отчёта и метрики.

### 11.3. Версия отчёта

В `summary.json` и MLflow сохраняются:

* `report_version` — версия методики/отчёта (`2.2.0`);
* `config_hash` — хэш параметров и разрешённых шаблонов;
* `template_fingerprint` — отдельный отпечаток шаблонов;
* `corpus_fingerprint` — отпечаток `assignments.jsonl`, `submissions.jsonl`,
  `pairs.jsonl`.

Поэтому по артефактам видно, какая именно конфигурация дала какие метрики.
Изменение шаблона или параметров меняет `config_hash`; изменение корпуса —
`corpus_fingerprint`.

### 11.4. Почему score не доказывает нарушение

Score — это мера лексического/векторного сходства, а не доказательство
авторства или умысла. Одинаковые имена стандартных функций, общая структура
задачи, разрешённый шаблон и служебные n-граммы поднимают score у независимых
работ. Поэтому результат обязан вести к ручной проверке, а `derivation_relation`
в корпусе — это происхождение примера, не вердикт.

Возможные утечки и что с ними сделано:

* TF-IDF обучается только на tuning, evaluation только `transform`;
* split идёт по `family_id`, семьи не пересекают границу tuning/evaluation;
* порог выбирается только на tuning, evaluation не участвует в выборе;
* `reference_label` не входит в признаки ML; `template_matched_lines`
  вычисляется по коду и шаблону, а не берётся из разметки;
* self-match и работы того же студента исключены;
* ошибки на tuning показаны для диагностики, итоговые метрики — на evaluation.

### 11.5. Какие выводы нельзя делать по малому корпусу

40 пар, 18 tuning / 22 evaluation и 20 семейств — это учебная демонстрация.
Bootstrap F1 = 0.864 ± 0.047, grouped-CV F1 = 0.630 ± 0.457: разброс слишком
велик, чтобы заявлять устойчивое преимущество ML или готовность к реальным
решениям. Нельзя делать выводы о конкретном студенте, переносить метрики на
реальные работы или использовать score как основание для санкции.

### 11.6. Экспериментальный детектор ИИ

В текущей версии экспериментальный детектор ИИ не реализован и в принятии
решений не участвует. `AIDeclarationChecker` проверяет только согласованность
AI-декларации с объёмом работы и не определяет авторство. Если экспериментальный
детектор добавляется как углубление, он обязан иметь отдельную оценку ошибок и
отказов (false positive/false negative, `None`), оставаться advisory-сигналом и
не влиять на санкцию.
