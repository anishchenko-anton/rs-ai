# rag_system/models/embeddings.py

from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

class IEmbeddingModel(ABC):
    """
    Абстрактный базовый класс (интерфейс) для моделей эмбеддингов.
    Определяет метод, который должны реализовывать все конкретные модели.
    """
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Преобразует список текстовых строк в их векторные представления (эмбеддинги).
        :param texts: Список строк для кодирования.
        :return: NumPy массив эмбеддингов (каждая строка - это вектор).
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Возвращает размерность генерируемых эмбеддингов.
        :return: Целое число - размерность вектора.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Возвращает уникальное имя модели.
        :return: Строка - имя модели.
        """
        pass

class SentenceTransformerEmbeddingModel(IEmbeddingModel):
    def __init__(self, model_name: str):
        self._model_name = model_name
        print(f"Загрузка SentenceTransformer модели: {self._model_name}...")
        try:
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"Модель '{self._model_name}' успешно загружена. Размерность: {self._dimension}.")
        except Exception as e:
            print(f"Ошибка при загрузке SentenceTransformer модели '{self._model_name}': {e}")
            print("Пожалуйста, убедитесь, что у вас есть подключение к интернету для первой загрузки.")
            self._model = None
            self._dimension = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Модель эмбеддингов не загружена. Невозможно сгенерировать эмбеддинги.")
        return self._model.encode(texts, normalize_embeddings=True).astype('float32')

    def get_dimension(self) -> int:
        return self._dimension
    
    def get_name(self) -> str:
        return self._model_name