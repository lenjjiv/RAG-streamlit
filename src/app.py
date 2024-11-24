import os
import streamlit as st
from model import ChatModel
import rag_util

# Путь к директории для хранения загруженных файлов
FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

# Настройка заголовка приложения
st.title("Чат-бот с RAG поддержкой")


@st.cache_resource
def load_model():
    """
    Загрузка языковой модели с кэшированием.
    """
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    return model


@st.cache_resource
def load_encoder():
    """
    Загрузка модели для создания эмбеддингов с кэшированием.
    """
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder


# Загружаем модели один раз и кэшируем их
model = load_model()
encoder = load_encoder()


def save_file(uploaded_file):
    """
    Сохранение загруженного файла на диск.
    
    Args:
        uploaded_file: Загруженный через Streamlit файл
        
    Returns:
        str: Путь к сохраненному файлу
    """
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Настройка боковой панели
with st.sidebar:
    # Параметры генерации и поиска
    max_new_tokens = st.number_input("Максимальное количество токенов в ответе", 128, 4096, 512)
    k = st.number_input("Количество документов для контекста", 1, 10, 3)
    
    # Загрузка PDF файлов
    uploaded_files = st.file_uploader(
        "Загрузите PDF файлы для контекста", type=["PDF", "pdf"], accept_multiple_files=True
    )
    
    # Обработка загруженных файлов
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)


# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Обработка пользовательского ввода
if prompt := st.chat_input("Задайте вопрос..."):
    # Добавление сообщения пользователя в историю
    st.session_state.messages.append({"role": "user", "content