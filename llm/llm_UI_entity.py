from prompt_template_utils import get_prompt_template
import torch
import streamlit as st
from run_localGPT import load_model
from langchain.callbacks.base import BaseCallbackHandler

equipment_question="Выведите без заголовка только измерительное оборудование (аппаратуру, приспособления, приборы, машины), представленное в Context, с названием раздела или параграфа, в котором оно было найдено (например, 4.1). Если в контексте оборудование не найдено, ответьте пустой строкой."
promptOuter,_ = get_prompt_template()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + "/"
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

# Sidebar contents
with st.sidebar:
    st.title("💬 Автоматизация стандартов ")
    st.markdown(
        """
    ## Инфо
    Приложение основано на:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LocalGPT](https://github.com/PromtEngineer/localGPT)
    """
    )


if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

chat_box = st.empty()
stream_handler = StreamHandler(chat_box, display_method='write')

if "LLM" not in st.session_state:
    LLM = load_model(stream_handler)
    st.session_state["LLM"] = LLM

st.title("Документы ближе 💬")
# Create a text input box for the user
prompt = st.text_area("Введите текст для извлечения оборудования по разделам", max_chars=2000, height=400)

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = st.session_state["LLM"](prompt=promptOuter.format(context=prompt, question=equipment_question))
    # ...and write it out to the screen
    st.write(response)
