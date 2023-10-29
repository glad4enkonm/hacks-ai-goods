import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
       file_extension = os.path.splitext(file_path)[1]
       loader_class = DOCUMENT_MAP.get(file_extension)
       if loader_class:
           file_log(file_path + ' loaded.')
           loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()[0]
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None 

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
           file_log(name + ' failed to submit')
           return None
        else:
           data_list = [future.result() for future in futures]
           # return data and file paths
           return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print('Importing: ' + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
               future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
               file_log('executor task failed: %s' % (ex))
               future = None
            if future is not None:
               futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log('Exception: %s' % (ex))
                
    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",          
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, _ = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=880, chunk_overlap=200)    
    texts = text_splitter.split_documents(text_documents)
    
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
   


def mainProcess(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, _ = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)    
    texts = text_splitter.split_documents(text_documents)
    
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # get the prompt template and memory if set by the user.
    prompt_template, _ = get_prompt_template(promptTemplate_type="llama", history=False)

    # load the llm pipeline
    #llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    llm = Ollama(model="llama2:70b-chat",
            temperature=0,
            top_k=1,
            top_p=0.1,
            verbose=True,
            repeat_penalty=1.3,
            callback_manager=callback_manager)
    
    equipment_question="Выведите без заголовка только измерительное оборудование (аппаратуру, приспособления, приборы, машины), представленное в Context, с названием раздела или параграфа, в котором оно было найдено (например, 4.1). Если в контексте оборудование не найдено, ответьте пустой строкой."
    
    len_chunks = len(texts)
    for idx, chunk in enumerate(texts):
        logging.info(f"{idx}/{len_chunks} chunk processing")
        llm_answer = llm(prompt=prompt_template.format(context=chunk, question=equipment_question))
        logging.info(llm_answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
