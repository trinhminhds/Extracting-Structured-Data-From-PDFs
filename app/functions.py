from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
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
        filename(str): The cleaned filename.
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
        temp_file = tempfile.NamedTemporaryFile(delete = False)
        temp_file.write(input_pdf)
        temp_file.close()

        # Load the PDF document
        loader = PyMuPDFLoader(temp_file.name)
        documents = loader.load()

        return documents

    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)


def split_document(documents, chunk_size, chunk_overlap):
    """
        Function to split generic text into smaller chunks.
        chunk_size: The desired maximum size of each chunk (default: 400)
        chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20

        Returns:
            list: A list of smaller text chunks created from the generic text.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap,
                                                   length_function = len, separators = ["\n\n", "\n", " "])

    # Join the list of documents into a single string
    text = "\n\n".join(doc.page_content for doc in documents)

    return text_splitter.split_text(text)


def get_embeddings_function(api_key):
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


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path = "db"):
    """
    Create a vector store from a list of text chunks.

    Parameters:
        chunks: A list of generic text chunks
        embedding_function: A function that takes a string and returns a vector
        file_name: The name of the file to associate with the vector store
        vector_store_path: The directory to store the vector store

    Returns: A Chroma vector store object
    """

    # Create a list of unique ids for each chunk based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)) for chunk in chunks]

    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    # Create a new Chroma database from the chunks
    vectorstore = Chroma.from_texts(texts = unique_chunks, collection_name = clean_filename(file_name),
                                    embedding = embedding_function, ids = list(unique_ids),
                                    persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # But we can also force it to save using the persist() method
    vectorstore.persist()
    return vectorstore


def create_vectorstore_from_texts(documents, api_key, file_name):
    # Step 2: Split the documents
    """
    Create a vector store from a list of texts

    Parameters:
        documents: A list of text documents
        api_key: The OpenAI API key used to create the vector store
        file_name: The name of the file to associate with the vector store

    Returns: A Chroma vector store object
    """
    docs = split_document(documents, chunk_size = 1000, chunk_overlap = 200)

    # Step 3: define embedding function
    embedding_function = get_embeddings_function(api_key)

    # Step 4: Create the vector store
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    return vectorstore


def load_vectorstore(file_name, api_key, vectorstore_path = "db"):
    """
    Load a previously saved Chroma vector store from disk.

    Parameters:
        file_name: The name of the file to load (without the path)
        api_key: The OpenAI API key used to create the vector store
        vectorstore_path: The path to the directory where the vector store was saved (default: "db")

    Returns: A Chroma vector store object
    """
    embedding_function = get_embeddings_function(api_key)
    return Chroma(persist_directory = vectorstore_path, embedding_function = embedding_function,
                  collection_name = clean_filename(file_name))


# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the questions. 
If you don't know the answer, say that you don't know. DON'T MAKE UP ANSWERS.
{context}
"""


class AnsweWithSource(BaseModel):
    """
    An answer to the question, with sources and reasoning.
    """
    answer: str = Field(description = "Answer to question.")
    source: str = Field(description = "Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description = "Explain the reasoning of the answer on the source")


class ExtractedInfoWithSources(BaseModel):
    """
    Extracted information about the research article.
    """
    paper_title: AnsweWithSource
    paper_summary: AnsweWithSource
    publication_year: AnsweWithSource
    paper_authors: AnsweWithSource


def format_docs(docs):
    """
    Format a list if Document objects as a single string.

    Parameters:
        docs(list): A list of Document objects.

    Returns:
        str: A string containing the text of all the documents joined by two newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings
# RunnablePassthrough() passes the input question unchanged.
def query_document(vectorstore, query, api_key):
    """
    Query a vector store with a question and return a structured response.

    Parameters:
        vectorstore: A Chroma vector store object.
        query: The question to ask the vector store.
        api_key: The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        A pandas DataFrame with three row: 'answer', 'source', 'reasoning'
    """

    llm = ChatOpenAI(model = "gpt-4o-mini", api_key = api_key)

    retriever = vectorstore.as_retriever(search_type_ = 'similarity')

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    reg_chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()} | prompt_template | llm.with_structured_output(ExtractedInfoWithSources,
                                                                                               strict = True)
    )

    structured_output = reg_chain.invoke(query)
    df = pd.DataFrame([structured_output.dict()])

    # Transforming into a table with two rows: 'answer' and 'source'
    answer_row = []
    source_row = []
    reasoning_row = []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['source'])
        reasoning_row.append(df[col][0]['reasoning'])

    # Create new dataframe with two rows: 'answer' and 'source'
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns = df.columns,
                                          index = ['answer', 'source', 'reasoning'])
    return structured_response_df.T
