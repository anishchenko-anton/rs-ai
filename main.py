# main.py

import json
import os
import sys

# current_dir - это директория, где находится main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)
# Импортируем классы и константы из наших модулей
from rag_system.config import (
    KNOWLEDGE_BASE_JSON_PATH,
    MAIN_RETRIEVER_MODEL,
    SECONDARY_RETRIEVER_MODEL,
    DATA_DIR
)
from rag_system.core.managers import EmbeddingModelManager, KnowledgeBaseManager

def main():
    # --- 1. Инициализация системы ретривера (выполняется ОДИН РАЗ при запуске приложения) ---
    print("--- Инициализация системы ретривера ---")

    print(f"Попытка загрузить данные из '{KNOWLEDGE_BASE_JSON_PATH}'...")
    try:
        with open(KNOWLEDGE_BASE_JSON_PATH, 'r', encoding='utf-8') as f:
            # Предполагаем, что JSON-файл содержит ключ "documents", как и наш фиктивный
            loaded_data = json.load(f)
            if "documents" in loaded_data:
                knowledge_base_data = loaded_data["documents"]
            else:
                # Если ваш JSON просто список объектов, то так:
                knowledge_base_data = loaded_data
        print(f"Успешно загружено {len(knowledge_base_data)} документов из '{KNOWLEDGE_BASE_JSON_PATH}'.")
    except FileNotFoundError:
        print(f"Ошибка: Файл '{KNOWLEDGE_BASE_JSON_PATH}' не найден. Убедитесь, что путь правильный и файл существует.")
        return # Выходим, если файл не найден
    except json.JSONDecodeError:
        print(f"Ошибка: Невозможно декодировать JSON из файла '{KNOWLEDGE_BASE_JSON_PATH}'. Проверьте формат файла.")
        return # Выходим, если файл некорректен
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при загрузке данных: {e}")
        return

    # 1.2. Создаем менеджер моделей эмбеддингов
    embedding_model_manager = EmbeddingModelManager()

    # 1.3. Создаем менеджер баз знаний (который использует менеджер моделей)
    knowledge_base_manager = KnowledgeBaseManager(embedding_model_manager)

    # 1.4. Получаем ретриверы для каждой модели и индексируем данные
    # Это действие также загружает/создает индексы FAISS и сохраняет их

    # Для MAIN_RETRIEVER_MODEL
    main_retriever = knowledge_base_manager.get_retriever(MAIN_RETRIEVER_MODEL)
    if main_retriever.index is None or len(main_retriever.documents_meta) == 0:
        print(f"\nИндекс для '{MAIN_RETRIEVER_MODEL}' пуст или не удалось загрузить. Добавляем документы...")
        # !!! ИСПРАВЛЕНИЕ ЗДЕСЬ: Используем уже загруженные knowledge_base_data !!!
        main_retriever.add_documents_to_index(knowledge_base_data)
    else:
        print(f"\nИндекс для '{MAIN_RETRIEVER_MODEL}' уже содержит документы, пропускаем добавление.")

    # Для SECONDARY_RETRIEVER_MODEL
    secondary_retriever = knowledge_base_manager.get_retriever(SECONDARY_RETRIEVER_MODEL)
    if secondary_retriever.index is None or len(secondary_retriever.documents_meta) == 0:
        print(f"\nИндекс для '{SECONDARY_RETRIEVER_MODEL}' пуст или не удалось загрузить. Добавляем документы...")
        # !!! ИСПРАВЛЕНИЕ ЗДЕСЬ: Используем уже загруженные knowledge_base_data !!!
        secondary_retriever.add_documents_to_index(knowledge_base_data)
    else:
        print(f"\nИндекс для '{SECONDARY_RETRIEVER_MODEL}' уже содержит документы, пропускаем добавление.")

    print("\n--- Система ретривера готова к работе! ---")

    # --- 2. Пример интерактивного использования (имитация запросов пользователя) ---
    print("\nВведите запросы для поиска тренировок. Введите 'exit' для выхода.")
    while True:
        user_query = input("\nВаш запрос (русский): ")
        if user_query.lower() == 'exit':
            break

        # --- Здесь мы имитируем получение контекста пользователя из БД ---
        user_goal = "набор массы"
        print(f" (Имитация: Цель пользователя - '{user_goal}')")

        # --- Выбор ретривера для текущего запроса ---
        if "yoga" in user_query.lower() or "english" in user_query.lower() or "flexibility" in user_query.lower():
            current_retriever = secondary_retriever
            print(f" Используем ретривер на базе модели: {SECONDARY_RETRIEVER_MODEL}")
        else:
            current_retriever = main_retriever
            print(f" Используем ретривер на базе модели: {MAIN_RETRIEVER_MODEL}")

        # --- Выполнение Retrieval (Поиск релевантных документов) ---
        retrieved_documents = current_retriever.retrieve(user_query, top_k=3)

        print("\n--- Найденные релевантные тренировки (для контекста LLM) ---")
        if retrieved_documents:
            for doc in retrieved_documents:
                print(f"  ID: {doc['id']}, Оценка: {doc['score']:.4f}, Текст: {doc['text']}")
        else:
            print("  Не найдено релевантных тренировок.")

        # --- Здесь должна быть интеграция с большой LLM ---
        # response_from_llm = your_llm_api_client.generate_response(...)
        # print("\n--- Рекомендация LLM (будет тут) ---")
        # print(response_from_llm)

if __name__ == "__main__":
    main()