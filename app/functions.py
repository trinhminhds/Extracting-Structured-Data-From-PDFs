from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re


def clean_filename(filename):
    """
    Remove '(number)' pattern from filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        str: The cleaned filename. 
    """
    # Regular expression to remove '(number)' pattern
    new_filename = re.sub(r'\(\d+\)', '', filename)
    return new_filename


def get_pdf_text(uploaded_file):
    """"
    Load a PDF document from an uploaded file and return it as a list of documents.
    
    Parameters:
        uploaded_file(file-like object): The uploaded PDF file to load.
    
    Returns:
        list: A list of documents extracted from the PDF file.
    """
    try:
        # Read file content
        input_pdf = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamline's file_uploader)
        tempfile = tempfile.NamedTemporaryFile(delete = False)
        tempfile.write(input_pdf)
        tempfile.close()

        # Load the PDF document
        loader = PyMuPDFLoader(tempfile.name)
        documents = loader.load()

        return documents

    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(tempfile.name)


def split_document(documents, chunk_size, chunk_overlap):
    """
        Function to split generic text into smaller chunks.
        chunk_size: The desired maximum size of each chunk (default: 400)
        chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20

        Returns:
            list: A list of smaller text chunks created from the generic text.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap,
                                                   lenght_function = len, separators = ["\n\n", "\n", " "])

    return text_splitter.split_text(documents)


def get_embeddings(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vectors embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided as an argument to the function.

    Parameters:
        api_key(str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002", api_key = api_key)
    return embeddings


def create_vectorstore(chunks, embeddings_function, file_name, vector_store_path = "db"):
    """

    """



















