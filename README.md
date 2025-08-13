# SpamDetection_Tests
Решение задачи бинарной классификации (spam = 1 / ham = 0) с использованием методов NLP — векторизация текстов + классические классификаторы. Ветка configs имеет переработанную структуру: конфиги в YAML, разделение поиска гиперпараметров, финальной тренировки и валидации, удобный экспорт артефактов.

## Datasets

Для обучения были выбраны два популярных набора данных с Kaggle, соответствующих заданию: единый CSV-файл со строковыми сообщениями и бинарной переменной классификации.
- **The Enron Email Dataset** [https://www.kaggle.com/datasets/mohinurabdurahimova/maildataset]
- **Ling-Spam Dataset** [https://www.kaggle.com/datasets/mandygu/lingspam-dataset]
- **Валидационный набор писем** [https://disk.yandex.ru/d/mJ_9zG3dX3eZgg]

В коде данные загружаются/сливаются в единый DataFrame (есть утилиты для скачивания и объединения).

## Алгоритм
Базовая идея: преобразовать письмо в вектор (CountVectorizer / TfidfVectorizer / SBERT / FastText и т.п.), затем применить бинарный классификатор (MultinomialNB, LogisticRegression и др.).

Для поиска гиперпараметров используется GridSearchCV (метрика оптимизации — f1_macro по умолчанию).

Workflow разделён: сначала оптимизация гиперпараметров (сохраняется results.json); затем финальная тренировка выбранных конфигураций на полном обучающем наборе;
Отдельно происходит валидация и измерение inference-метрик (latency / throughput / model size / accuracy/AUC).

## Структура проекта

- scripts
  - run_experiment.py — запуск поиска (CV) по YAML-конфигу; сохраняет experiments/<name>/results.json.
  - train_pipes.py — берет results.json, восстанавливает pipeline и обучает его на полном наборе; сохраняет experiments/<name>/final/best_pipeline.joblib и final/train_metrics.json.
  - validate_pipes.py — загружает финальные пайплайны и измеряет inference-метрики; сохраняет final/inference_metrics.json.
- src
  - config.py — парсер нового YAML-формата (вложенные блоки vectorizer и classifier).
  - model_builder.py — registry векторизаторов/классификаторов, сборка пайплайна, создание param_grid и реконструкция pipeline из сохранённых params.
  - trainer.py — обёртки обучения/оценки, измерения времени и размера модели.
  - utils.py — сохранение артефактов, сбор сводных таблиц, утилиты для валидации/измерений.
  - vectorizers — папка для локальных имплементаций различных векторизаций.
  - data_loader.py — утилиты для скачивания и подготовки данных.
- experiments — папка для хранения полных экспериментов (конфигурация, результаты, модели).

## Формат файла конфигурации эксперимента

    #--------------------- DATASET ---------------------
    
    datasets: [enron, lingspam]
    sample_size: 1
    
    test_size: 0.2   # fraction of the data kept for *validation*
    
    #--------------------- MODEL ----------------------
    
    vectorizer:
      TfidfVectorizer:
        min_df:
          - 1
          - 3
        ngram_range:
          - [1, 1]
          - [1, 2]
    
    
    classifier:
      LinearSVC:
        C:
          - 1
          - 5
        tol:
          - 0.001
          - 0.0001
        loss:
          - squared_hinge
    
    #--------------------- TRAINING --------------------
    cv_folds: 3
    scoring: f1_macro                 # metric to optimize
    random_state: 42                    # reproducibility seed

Все параметры, следующие за типом векторизатора/классификатора должны быть списками (для корректной работы GridsearchCV).
