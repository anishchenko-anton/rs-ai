# rag_system/core/retriever.py

import json
import os
import faiss
import numpy as np

from rag_system.models.embeddings import IEmbeddingModel # Импортируем интерфейс модели
from rag_system.config import BASE_INDEX_DIR             # Импортируем константу из config.py

class LocalKnowledgeBaseRetriever:
    def __init__(self, embedding_model: IEmbeddingModel): # Принимает уже загруженную модель
        self.embedding_model = embedding_model
        self.model_name = self.embedding_model.get_name()
        
        # Пути к файлам индекса и метаданных для этой конкретной модели
        # Заменяем символы, которые могут быть проблемой в именах файлов
        safe_model_name = self.model_name.replace('/', '_').replace('-', '_')
        self.model_index_dir = os.path.join(BASE_INDEX_DIR, safe_model_name)
        self.faiss_index_path = os.path.join(self.model_index_dir, 'faiss_index.bin')
        self.documents_meta_path = os.path.join(self.model_index_dir, 'documents_meta.json')
        
        self.index = None
        self.documents_meta = []
        
        # Убедимся, что директория для индекса существует
        os.makedirs(self.model_index_dir, exist_ok=True)

        # Попытка загрузить существующий индекс и метаданные
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.documents_meta_path):
            print(f"Обнаружены существующие файлы индекса и метаданных для '{self.model_name}'. Попытка загрузки...")
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                with open(self.documents_meta_path, 'r', encoding='utf-8') as f:
                    self.documents_meta = json.load(f)
                print(f"Индекс FAISS и метаданные для '{self.model_name}' успешно загружены. Документов: {len(self.documents_meta)}")
            except Exception as e:
                print(f"Ошибка при загрузке индекса/метаданных для '{self.model_name}': {e}. Будет создан новый индекс.")
                self.index = None
                self.documents_meta = []
        else:
            print(f"Существующие файлы индекса/метаданных для '{self.model_name}' не найдены. Будет создан новый индекс.")

    def add_documents_to_index(self, documents: list[dict]):
        """
        Добавляет новые документы в индекс.
        documents: список словарей, где каждый словарь {'id': int, 'text': str}
        """
        if self.embedding_model.get_dimension() == 0:
            print("Модель эмбеддингов не загружена, невозможно добавить документы.")
            return

        print(f"Добавление {len(documents)} документов в индекс для модели '{self.model_name}'...")
        new_texts = [doc['text'] for doc in documents]
        
        try:
            new_embeddings = self.embedding_model.encode(new_texts)
        except RuntimeError as e:
            print(f"Ошибка при генерации эмбеддингов: {e}")
            return

        if not new_embeddings.size:
            print("Не удалось сгенерировать эмбеддинги для новых документов.")
            return

        if self.index is None:
            dimension = self.embedding_model.get_dimension()
            self.index = faiss.IndexFlatIP(dimension)
            print(f"Новый индекс FAISS инициализирован для '{self.model_name}' с размерностью {dimension}.")
        
        self.index.add(new_embeddings)
        
        start_id = len(self.documents_meta)
        for i, doc in enumerate(documents):
            if 'id' not in doc or doc['id'] is None:
                doc['id'] = start_id + i 
            self.documents_meta.append({'id': doc['id'], 'text': doc['text']})
        
        print(f"Документы добавлены. Общее количество документов в индексе: {len(self.documents_meta)}")
        self._save_index_and_meta()

    def _save_index_and_meta(self):
        """Сохраняет FAISS индекс и метаданные документов на диск для текущей модели."""
        if self.index is not None:
            faiss.write_index(self.index, self.faiss_index_path)
            print(f"Индекс FAISS для '{self.model_name}' сохранен в '{self.faiss_index_path}'")
        
        with open(self.documents_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents_meta, f, ensure_ascii=False, indent=4)
        print(f"Метаданные документов для '{self.model_name}' сохранены в '{self.documents_meta_path}'")

    def retrieve(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Выполняет поиск релевантных документов по запросу.
        :param query_text: Текст запроса пользователя (может быть на любом поддерживаемом языке).
        :param top_k: Количество наиболее релевантных документов для возврата.
        :return: Список словарей с найденными документами ('id', 'text', 'score').
        """
        if self.index is None or not self.documents_meta or self.embedding_model.get_dimension() == 0:
            print("Индекс не инициализирован, база знаний пуста или модель эмбеддингов не загружена. Невозможно выполнить поиск.")
            return []

        print(f"Поиск релевантных документов для запроса: '{query_text}' (Модель: {self.model_name})")
        
        try:
            query_embedding = self.embedding_model.encode([query_text])
        except RuntimeError as e:
            print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
            return []

        if not query_embedding.size:
             print("Не удалось сгенерировать эмбеддинг для запроса.")
             return []

        D, I = self.index.search(query_embedding, top_k)
        
        retrieved_documents = []
        for i, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.documents_meta):
                doc_score = D[0][i] 
                doc_text = self.documents_meta[idx]['text']
                doc_id = self.documents_meta[idx]['id'] 

                retrieved_documents.append({
                    'id': doc_id,
                    'text': doc_text,
                    'score': float(doc_score)
                })
                print(f"  - Найден документ (Score: {doc_score:.4f}): '{doc_text[:100]}...'") 

        return retrieved_documents