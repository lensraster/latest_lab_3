# Звіт про запуск та результати

## 1. Інформація про роботу

- **Дисципліна:** Основи обробки природної мови
- **Тема:** Донавчання варіанту BERT для класифікації тексту
- **Модель:** `bert-base-uncased`
- **Датасет:** `ag_news`

## 2. Підготовка середовища

Нижче наведено команди для підготовки:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Результати тестування

Команда:

```bash
pytest -q
```

Результат фактичного запуску:

- `8 passed in 5.94s`
- Усі тести завершилися успішно.

## 4. Запуск донавчання

Команда:

```bash
python -m nlp_bert_classifier.cli train \
  --train-samples 200 \
  --test-samples 80 \
  --epochs 1 \
  --batch-size 8
```

Результат фактичного запуску (ключові метрики):

```json
{
  "eval_loss": 1.146956443786621,
  "eval_accuracy": 0.6,
  "eval_f1_weighted": 0.5798051948051948,
  "eval_runtime": 0.2046,
  "eval_samples_per_second": 390.916,
  "eval_steps_per_second": 48.864,
  "epoch": 1.0
}
```

Додаткові артефакти:

- `artifacts/bert-ag-news/model/` — збережена донавчена модель;
- `artifacts/bert-ag-news/metrics.json` — метрики та конфігурація експерименту.

## 5. Запуск інференсу

Команда:

```bash
python -m nlp_bert_classifier.cli predict \
  --model-dir artifacts/bert-ag-news/model \
  --text "Apple releases a new AI chip for data centers"
```

Результат фактичного запуску:

```json
{
  "label": "Business",
  "confidence": 0.32391735911369324
}
```

## 6. Другий експеримент (порівняльний)

Мета: перевірити іншу BERT-сумісну модель за тих самих налаштувань вибірки та епох.

Команда:

```bash
python -m nlp_bert_classifier.cli train \
  --model-name distilbert-base-uncased \
  --output-dir artifacts/distilbert-ag-news \
  --train-samples 200 \
  --test-samples 80 \
  --epochs 1 \
  --batch-size 8
```

Результат фактичного запуску (ключові метрики):

```json
{
  "eval_loss": 1.3238270282745361,
  "eval_accuracy": 0.275,
  "eval_f1_weighted": 0.1717266775777414,
  "eval_runtime": 0.1221,
  "eval_samples_per_second": 655.26,
  "eval_steps_per_second": 81.908,
  "epoch": 1.0
}
```

Порівняння з базовим експериментом:

- `bert-base-uncased`: `eval_accuracy = 0.60`, `eval_f1_weighted = 0.5798`;
- `distilbert-base-uncased`: `eval_accuracy = 0.275`, `eval_f1_weighted = 0.1717`.

Висновок: для поточних умов експерименту (невелика навчальна підвибірка та 1 епоха) кращу якість показала модель `bert-base-uncased`.

## 7. Підсумковий висновок

Поставлена навчальна задача виконана: обрано та донавчено варіант BERT (`bert-base-uncased`) для класифікації тексту на `ag_news`, виконано інференс і проведено додатковий порівняльний експеримент (`distilbert-base-uncased`). Автоматизовані тести пройдено успішно, проєкт готовий до демонстрації й оцінювання.
