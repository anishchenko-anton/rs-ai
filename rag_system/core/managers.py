# core/managers.py

from rag_system.models.embeddings import IEmbeddingModel, SentenceTransformerEmbeddingModel # Импортируем конкретную модель
from rag_system.core.retriever import LocalKnowledgeBaseRetriever                      # Импортируем ретривер

class EmbeddingModelManager:
    def __init__(self):
        self._models: dict[str, IEmbeddingModel] = {} # Словарь для хранения загруженных моделей

    def load_model(self, model_name: str) -> IEmbeddingModel:
        """
        Загружает модель эмбеддингов по имени. Если модель уже загружена, возвращает её.
        :param model_name: Имя модели для загрузки/получения.
        :return: Экземпляр IEmbeddingModel.
        """
        if model_name not in self._models:
            # Здесь мы можем добавить логику для разных типов моделей,
            # но пока используем только SentenceTransformer
            if model_name.startswith("paraphrase-") or "MiniLM-L" in model_name: # Более общее условие
                self._models[model_name] = SentenceTransformerEmbeddingModel(model_name)
            else:
                # В будущем можно добавить поддержку других типов моделей
                raise ValueError(f"Неизвестный тип модели: {model_name}. Добавьте реализацию.")
        
        # Проверка, успешно ли загрузилась модель
        if self._models[model_name].get_dimension() == 0:
            print(f"Предупреждение: Модель '{model_name}' не смогла быть загружена.")
            del self._models[model_name] # Удаляем неудачно загруженную модель
            raise RuntimeError(f"Модель '{model_name}' не загружена, проверьте логи.")
            
        return self._models[model_name]

    def get_model(self, model_name: str) -> IEmbeddingModel:
        """
        Возвращает загруженную модель по имени. Вызывает ошибку, если модель не загружена.
        """
        if model_name not in self._models:
            raise ValueError(f"Модель '{model_name}' не загружена. Используйте load_model() сначала.")
        return self._models[model_name]

class KnowledgeBaseManager:
    def __init__(self, embedding_model_manager: EmbeddingModelManager):
        self.embedding_model_manager = embedding_model_manager
        self._retrievers: dict[str, LocalKnowledgeBaseRetriever] = {}

    def get_retriever(self, model_name: str) -> LocalKnowledgeBaseRetriever:
        """
        Возвращает ретривер для указанной модели.
        Если ретривер еще не создан, он будет инициализирован и загружен.
        """
        if model_name not in self._retrievers:
            print(f"Инициализация ретривера для модели: {model_name}...")
            # Получаем модель эмбеддингов из менеджера
            embedding_model_instance = self.embedding_model_manager.load_model(model_name)
            # Передаем загруженную модель ретриверу
            self._retrievers[model_name] = LocalKnowledgeBaseRetriever(embedding_model_instance)
        return self._retrievers[model_name]