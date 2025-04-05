from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from google import genai
import os
import shutil
# from IPython.display import display, Markdown

#####


# function to load specific pdf file
def load_pdf_document(path):
    loader = PyPDFLoader(path)
    return loader.load()


# function to load all pdf files in a directory
def load_pdf_directory(path):
    loader = PyPDFDirectoryLoader(path)
    return loader.load()


# function to load a webpage
def load_webpage(url):
    loader = WebBaseLoader(web_paths=(url,))
    return loader.load()


# load the pdfs
documents_path = "policy_documents"
documents = load_pdf_directory(documents_path)


# create a text splitter function
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


# split all the documents
chunks = split_documents(documents)

# load key
with open("api_keys/OpenAI_API_Key.txt", "r") as file:
    OPENAI_API_KEY = file.read().strip()

# declare db directory
chroma_path = "chroma_db"

# declare embeding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# create function to create chroma db
def save_to_chroma_db(chunks: list[Document]):
    # clear database if it exists
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    chroma = Chroma.from_documents(
        documents=chunks, persist_directory=chroma_path, embedding=embedding_model
    )
    chroma.persist()
    print(
        f"Chroma had successfully created Vector database from {len(chunks)} chunks and stored at /{chroma_path}/."
    )


# create chroma db
save_to_chroma_db(chunks)
