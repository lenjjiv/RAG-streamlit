import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer


# Путь к директории для кэширования моделей
CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


class Encoder:
    """
    Класс для создания эмбеддингов текста с помощью предобученной модели.
    
    Использует sentence-transformers для преобразования текста в векторные представления.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    ):
        """
        Инициализация энкодера.

        Args:
            model_name (str): Имя модели из HuggingFace
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device},
        )


class FaissDb:
    """
    Класс для работы с векторной базой данных FAISS.
    
    Обеспечивает индексацию и поиск похожих документов на основе их векторных представлений.
    """

    def __init__(self, docs, embedding_function):
        """
        Инициализация векторной базы данных.

        Args:
            docs: Список документов для индексации
            embedding_function: Функция для создания эмбеддингов
        """
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )

    def similarity_search(self, question: str, k: int = 3):
        """
        Поиск документов, релевантных заданному вопросу.

        Args:
            question (str): Текст вопроса
            k (int): Количество возвращаемых документов

        Returns:
            str: Конкатенированный текст найденных документов
        """
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context


def load_and_split_pdfs(file_paths: list, chunk_size: int = 256):
    """
    Загрузка и разделение PDF-документов на небольшие чанки.

    Args:
        file_paths (list): Список путей к PDF файлам
        chunk_size (int): Размер чанка в токенах

    Returns:
        list: Список чанков документов
    """
    # Загружаем PDF файлы
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    # Инициализируем токенизатор для разделения текста
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),  # 10% перекрытие между чанками
        strip_whitespace=True,
    )
    
    # Разделяем документы на чанки
    docs = text_splitter.split_documents(pages)
    return docs