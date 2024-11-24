import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

# Путь к директории для кэширования моделей
CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


class ChatModel:
    """
    Класс для работы с языковой моделью в режиме чата.
    
    Обеспечивает загрузку модели, токенизацию и генерацию ответов
    с поддержкой контекстного окна для RAG.
    """

    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):
        """
        Инициализация модели чата.

        Args:
            model_id (str): Идентификатор модели в HuggingFace
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        # Получаем токен доступа из переменных окружения
        ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

        # Инициализируем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )

        # Настраиваем квантизацию для оптимизации памяти
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Загружаем модель с квантизацией
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()  # Переключаем в режим оценки
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        """
        Генерация ответа на вопрос с учетом контекста.

        Args:
            question (str): Вопрос пользователя
            context (str, optional): Контекст из базы знаний
            max_new_tokens (int): Максимальное количество токенов в ответе

        Returns:
            str: Сгенерированный ответ модели
        """
        # Формируем промпт в зависимости от наличия контекста
        if context == None or context == "":
            prompt = f"""Дай подробный ответ на следующий вопрос. Вопрос: {question}"""
        else:
            prompt = f"""Используя информацию из контекста, дай подробный ответ на вопрос.
Контекст: {context}.
Вопрос: {question}"""

        # Подготавливаем чат для модели
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Токенизируем входные данные
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Декодируем и очищаем ответ
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt):]  # Удаляем входной промпт из ответа
        response = response.replace("<eos>", "")  # Удаляем токен конца последовательности

        return response