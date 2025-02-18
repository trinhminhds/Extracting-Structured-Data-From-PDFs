
# Extracting Structured Data from PDFs

A Python AI project that leverages large language models (LLMs) to extract key information from PDF documents. This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system that processes unstructured PDF data—such as research papers—to extract structured data like titles, summaries, authors, and publication years. It also includes a Streamlit web interface and Docker containerization for easy deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Using Docker](#using-docker)
- [Project Structure](#project-structure)


## Overview

Inspired by the Ig Nobel Prizes, this project automates the extraction of structured information from research papers using AI. Instead of manually reading and extracting data, the application utilizes advanced techniques in document processing, text chunking, vector embeddings, and prompt engineering. The result is a reliable system that answers user queries based on the provided PDF documents, citing its sources and reasoning.

## Features

- **Document Processing**: Load and process PDF documents using `PyPDF`.
- **Text Chunking**: Split documents into manageable chunks using LangChain’s `RecursiveCharacterTextSplitter`.
- **Embedding Generation**: Convert text chunks into numerical vectors (embeddings) using OpenAI’s embedding model.
- **Vector Database**: Store and query embeddings with ChromaDB for efficient similarity searches.
- **Retrieval-Augmented Generation (RAG)**: Retrieve relevant document sections and generate structured responses with an LLM.
- **Structured Outputs**: Define desired output formats (e.g., title as string, year as integer) and receive responses in a structured format.
- **Web Interface**: A user-friendly interface built with Streamlit.
- **Dockerized Deployment**: Containerize the application for cross-platform compatibility and easy deployment.

## Technologies Used

- **Python 3.11**
- **LangChain**
- **OpenAI API (GPT-4 and embeddings)**
- **ChromaDB (Vector Database)**
- **PyPDF**
- **Pandas**
- **Streamlit**
- **Docker**

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key (you can sign up at [OpenAI Platform](https://platform.openai.com/))
- Docker (for containerization)

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/trinhminhds/Extracting-Structured-Data-From-PDFs
   cd Extracting-Structured-Data-From-PDFs
   ```

2. **Create and activate a virtual environment:**
  
    ```
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```

4. **Set up your environment variables:**

    Create a `.env` file in the project root and add your OpenAI API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage
### Running Locally
1. **Start the Streamlit web app:**
   ```
   streamlit run app/app.py
   ```
   
2. **Open your browser and navigate to http://localhost:8501 to interact with the app.**
   
3. **Run the Jupyter Notebook:**
If you prefer working in a notebook, open the notebooks/data_extraction.ipynb file using Jupyter Lab or Notebook:
    ```
    jupyter lab  # or jupyter notebook
    ```

## Using Docker
1. **Build the Docker image:**
    ```
    docker build -t streamlit-app .
    ```

2. **Run the Docker container:**
    ```
    docker run -p 8501:8501 streamlit-app
    ```
3. **Access the app at http://localhost:8501.**

## Project Structure

    ├── app/
    │   └── functions.py            
    |   └── streamlit_app.py        # Streamlit web app
    ├── data/
    │   └── [PDF documents]   # Sample PDFs for extraction
    ├── notebooks/
    │   └── data_extraction_llms.ipynb  # Jupyter Notebook for project demonstration
    ├── .env                  # Environment variables file (not tracked by Git)
    ├── Dockerfile            # Docker configuration file
    ├── requirements.txt      # Python dependencies
    └── README.md             # Project documentation

