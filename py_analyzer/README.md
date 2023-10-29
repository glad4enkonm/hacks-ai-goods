# hacks-ai-goods

1. Используем [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) модель чтобы считать эмбеддинги
2. Применяем метод cosine_similarity для оценки сходства между двумя векторами-эмбеддингами
3. Считаем наиболее сходные строки из столбцов: 'Группа продукции', 'Наименование продукции' из dataset.csv со стандартами из standarts.csv и standarts_list.csv используя метод cosine_similarity для оценки сходства между каждой строкой


