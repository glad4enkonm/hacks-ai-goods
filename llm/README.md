# llm: Говорящие документы :speech_balloon:

**llm** делает понятнее документы, теперь им можно задать вопрос.

## Особенности :sparkles:
- **Локально**: все данные на своём сервере.
- **Большая моедль**: используется llama2:70b-chat модель, многое умеет многое понимает, приятно общаться :innocent:.
- **Таки не самая большая**: даже с настройкой на точность и консервативность модель не дообучалась, задачи задаются запросами так что бывают странности :alien:.
- **UI**: используется Streamlit для сайта, можно пообщаться в чате по документу.
- **API**: имеется для состыковки.
- **GPU, CPU**: модель считается на видеокарте через ollama, на сервере установлена GPU TeslaV100, embeddings сейчас считаются на CPU.


## Железо 🛠️
Сервер с GPU TeslaV100 32GB, 64GB RAM


## Использовались :floppy_disk:
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace LLMs](https://huggingface.co/models)
- [InstructorEmbeddings](https://instructor-embedding.github.io/)
- [LLAMACPP](https://github.com/abetlen/llama-cpp-python)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.ai/)
- [localGPT](https://github.com/PromtEngineer/localGPT)

# Настройка окружения 🌍

1. :electric_plug: git clone

```shell
git clone glad4enkonm/hacks-ai-goods.git
```

2. :rocket: Устанавливаем [conda](https://www.anaconda.com/download) 

```shell
conda create -n hackAiGoods python=3.10.0
conda activate hackAiGoods
```

3. 🛠️ подтягиваем зависимости

```shell
pip install -r requirements.txt
```

***Установка Ollama:***

[Инструкция на сайте](https://ollama.ai/)

### Поддерживаемые форматы:

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,    
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Загрузка данных

```shell
python ingest.py
```

### Web интерфейс

```shell
python llm_UI.py
```

```shell
python llm_UI_entity.py
```