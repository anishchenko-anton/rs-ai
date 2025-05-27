# config.py

import os

# --- Константы для моделей и путей к файлам ---

# Имя основной модели эмбеддингов для ретривера
MAIN_RETRIEVER_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
# MAIN_RETRIEVER_MODEL = 'paraphrase-MiniLM-L6-v2'

# Имя дополнительной модели эмбеддингов (для демонстрации выбора)
SECONDARY_RETRIEVER_MODEL = 'paraphrase-MiniLM-L6-v2'

# Базовая директория для хранения FAISS индексов
BASE_INDEX_DIR = 'faiss_indexes'

# Путь к файлу с исходной базой знаний (JSON)
KNOWLEDGE_BASE_JSON_PATH = 'rag_system/data/export_2025-05-27_15 01 35.json'

# Директория для данных
DATA_DIR = 'rag_system/data'

# Убедимся, что директория для данных существует
os.makedirs(DATA_DIR, exist_ok=True)