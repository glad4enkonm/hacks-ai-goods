from prompt_template_utils import get_prompt_template
import torch
import streamlit as st
from llm.run_llm import load_model
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA


# Sidebar contents
with st.sidebar:
    st.title("💬 Сделаем стандрты понятнее")
    st.markdown(
        """
    ## Инфо
    Приложение основано на:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LocalGPT](https://github.com/PromtEngineer/localGPT)
    """
    )


if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"



# Define the retreiver
# load the vectorstore
if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    RETRIEVER = DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    LLM = load_model()
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = get_prompt_template()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state["QA"] = QA

st.title("Документы ближе 💬")
# Create a text input box for the user
prompt = st.text_input("Введите запрос")

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    answer, docs = response["result"], response["source_documents"]
    # ...and write it out to the screen
    st.write(answer)

    # With a streamlit expander
    with st.expander("Похожие документы"):
        # Find the relevant pages
        search = st.session_state.DB.similarity_search_with_score(prompt)
        # Write out the first
        for i, doc in enumerate(search):
            # print(doc)
            st.write(f"Источник # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
            st.write(doc[0].page_content)
            st.write("--------------------------------")
