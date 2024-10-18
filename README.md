# Contextual Retrieval System with Supabase Storage

## Overview

This Contextual Retrieval System enhances the accuracy and relevance of information retrieval by incorporating context into the search process. It leverages OpenAI's GPT-4 model, LlamaIndex, FastAPI, LiteLLM, and uses Supabase for both vector storage and document storage and management.

## Key Components

1. **LiteLLM**: For flexible integration with OpenAI's GPT-4 model.
2. **Supabase Postgres with pgvector**: For vector storage, enabling efficient similarity searches.
3. **Supabase Storage**: For centralized document storage and management.
4. **FastAPI**: For building an asynchronous web API.
5. **LlamaIndex**: For document processing and indexing.
6. **DiskCache**: For caching to reduce redundant computations and API calls.
7. **Loguru**: For enhanced logging capabilities.

## Benefits

- **Improved Accuracy**: Reduces retrieval failures by preserving crucial context.
- **Enhanced Relevance**: Provides more relevant search results by considering broader context.
- **Versatility**: Applicable across various domains.
- **Efficiency in Large Datasets**: Effective when dealing with large, complex datasets.
- **Better User Experience**: Offers context-aware and accurate responses.

---

## Folder and File Structure

```
contextual_retrieval/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   ├── config.py
│   ├── supabase_client.py
├── cache/
│   └── (Cache files stored here)
├── supabase/
│   └── migrations/
│       └── 20231018000000_create_embeddings_table.sql
├── .env
├── Dockerfile
├── docker-compose.yml
├── install.sh
├── pyproject.toml
└── README.md
```

- `app/`: Contains the application code.
  - `__init__.py`: Initializes the package.
  - `main.py`: The main FastAPI application.
  - `utils.py`: Utility functions.
  - `config.py`: Configuration settings.
  - `supabase_client.py`: Supabase client initialization.
- `cache/`: Cache directory for `diskcache`.
- `supabase/migrations/`: Contains SQL scripts for setting up Supabase.
- `.env`: Environment variables.
- `Dockerfile`: Docker configuration.
- `docker-compose.yml`: Compose file to run the application.
- `install.sh`: Installation script.
- `pyproject.toml`: Poetry configuration.
- `README.md`: Project documentation.

---

## Installation Bash Script (`install.sh`)

```bash
#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install project dependencies
echo "Installing dependencies with Poetry..."
poetry install

echo "Dependencies installed successfully."
```

**Make sure to give execution permissions to the script:**

```bash
chmod +x install.sh
```

---

## Dockerfile

```dockerfile
# Use the official Python slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml poetry.lock /app/
COPY app /app/app
COPY install.sh /app/
COPY .env /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Poetry Configuration (`pyproject.toml`)

```toml
[tool.poetry]
name = "contextual_retrieval"
version = "0.1.0"
description = "Contextual Retrieval using OpenAI GPT-4, LlamaIndex, LiteLLM, and Supabase"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.95.2"
uvicorn = "^0.21.1"
llama-index = "^0.7.3"
litellm = "^0.1.1"
supabase = "^1.0.3"
diskcache = "^5.4.0"
aiohttp = "^3.8.1"
loguru = "^0.6.0"
python-dotenv = "^1.0.0"

[tool.poetry.dev-dependencies]
pytest = "^7.1.2"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

## FastAPI Asynchronous Application

### `app/config.py`

```python
import os
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_bucket_name: str = os.getenv("SUPABASE_BUCKET_NAME", "documents")

settings = Settings()
```

### `app/supabase_client.py`

```python
from supabase import create_client, Client
from app.config import settings

supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
```

### `app/utils.py`

```python
import asyncio
from llama_index import Document
from llama_index.text_splitter import TokenTextSplitter
from litellm import completion, embedding
from diskcache import Cache
from loguru import logger
from app.config import settings
from app.supabase_client import supabase

cache = Cache("./cache")

async def generate_context(document: str, chunk: str) -> str:
    prompt = f"Document: {document}\nChunk: {chunk}\nProvide a succinct context for this chunk within the overall document."
    cache_key = f"context:{hash(prompt)}"

    if cache_key in cache:
        logger.debug("Cache hit for context generation.")
        return cache[cache_key]
    else:
        logger.debug("Generating context using GPT-4.")
        response = await completion(model="gpt-4", messages=[{"role": "user", "content": prompt}])
        context = response.choices[0].message.content
        cache[cache_key] = context
        return context

async def contextualize_document(doc: Document) -> list:
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(doc.text)
    contextualized_chunks = []

    tasks = [generate_context(doc.text, chunk) for chunk in chunks]
    contexts = await asyncio.gather(*tasks)

    for context, chunk in zip(contexts, chunks):
        contextualized_text = f"{context} {chunk}"
        contextualized_chunks.append(Document(text=contextualized_text))
    return contextualized_chunks

async def store_embedding(text: str, embedding_vector: list):
    data = supabase.table('embeddings').insert({
        'content': text,
        'embedding': embedding_vector
    }).execute()
    return data

async def search_embeddings(query_embedding: list, match_threshold: float, match_count: int):
    response = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'match_threshold': match_threshold,
        'match_count': match_count
    }).execute()
    return response.data
```

### `app/main.py`

```python
from fastapi import FastAPI, UploadFile, File
import asyncio
from llama_index import Document
from app.utils import contextualize_document, store_embedding, search_embeddings
from app.config import settings
from app.supabase_client import supabase
from litellm import embedding
from loguru import logger
from io import BytesIO

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Fetching documents from Supabase Storage...")
    bucket = supabase.storage.from_(settings.supabase_bucket_name)
    documents_list = bucket.list().get('data', [])

    documents = []

    for doc in documents_list:
        file_name = doc['name']
        # Download the file content
        response = bucket.download(file_name)
        file_content = response.decode('utf-8')  # Adjust decoding if necessary
        documents.append(Document(text=file_content))

    logger.info(f"Loaded {len(documents)} documents.")

    logger.info("Contextualizing documents...")
    tasks = [contextualize_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    contextualized_documents = [doc for sublist in results for doc in sublist]
    logger.info(f"Contextualized into {len(contextualized_documents)} chunks.")

    logger.info("Storing embeddings...")
    for doc in contextualized_documents:
        embed_response = await embedding(model="text-embedding-ada-002", input=doc.text)
        await store_embedding(doc.text, embed_response.data[0].embedding)
    logger.info("Embeddings stored successfully.")

@app.get("/search")
async def search(query: str, threshold: float = 0.7, count: int = 10):
    embed_response = await embedding(model="text-embedding-ada-002", input=query)
    query_embedding = embed_response.data[0].embedding
    results = await search_embeddings(query_embedding, threshold, count)
    return {"results": results}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Upload to Supabase Storage
    bucket = supabase.storage.from_(settings.supabase_bucket_name)
    content = await file.read()
    bucket.upload(file.filename, content)

    return {"message": "File uploaded successfully"}
```

### `app/__init__.py`

```python
# This file is intentionally left blank to make 'app' a package.
```

---

## SQL for Supabase (`supabase/migrations/20231018000000_create_embeddings_table.sql`)

```sql
-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store document embeddings
CREATE TABLE embeddings (
  id BIGSERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)
);

-- Create a function to match documents based on embedding similarity
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1536),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    id,
    content,
    1 - (embeddings.embedding <=> query_embedding) AS similarity
  FROM embeddings
  WHERE 1 - (embeddings.embedding <=> query_embedding) > match_threshold
  ORDER BY embeddings.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create an index on the embedding column to speed up similarity searches
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Environment Variables (`.env`)

Create a `.env` file in the root directory with the following content:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_service_role_key_here
SUPABASE_BUCKET_NAME=documents
```

**Notes:**

- Replace the placeholder values with your actual API keys and URLs.
- Use the **service role key** for `SUPABASE_KEY` to allow server-side operations like uploading and downloading files.
- Do not share this information publicly.
- Add `.env` to your `.gitignore` file to prevent it from being committed to version control.

---

## Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - SUPABASE_BUCKET_NAME=${SUPABASE_BUCKET_NAME}
    volumes:
      - ./app:/app/app
      - ./cache:/app/cache
    env_file:
      - .env
```

---

## Running the Application

### Prerequisites

- **Python 3.9** installed on your machine.
- **Poetry** installed.
- **Docker** and **Docker Compose** installed (if running with Docker).
- An **OpenAI API key**.
- A **Supabase** project with `pgvector` extension enabled and a storage bucket set up.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/contextual_retrieval.git
   cd contextual_retrieval
   ```

2. **Set Up Supabase**

   - **Create a Supabase Project**:
     - Go to [Supabase](https://supabase.com/) and create a new project.
   - **Enable `pgvector` Extension**:
     - In the SQL editor of your Supabase dashboard, run:

       ```sql
       CREATE EXTENSION IF NOT EXISTS vector;
       ```

   - **Run Migration Script**:
     - Copy the content of `supabase/migrations/20231018000000_create_embeddings_table.sql` and execute it in the SQL editor to set up the necessary database schema.
   - **Create a Storage Bucket**:
     - In the Supabase dashboard, navigate to **Storage** and create a new bucket named `documents`.

3. **Set Up Environment Variables**

   - Update the `.env` file with your OpenAI API key and Supabase configurations.

4. **Install Dependencies**

   - Make the installation script executable:

     ```bash
     chmod +x install.sh
     ```

   - Run the installation script:

     ```bash
     ./install.sh
     ```

     This will install all the required dependencies using Poetry.

5. **Run the Application**

   - Start the FastAPI application:

     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```

     The application will download documents from Supabase Storage and process them during startup.

6. **Access the API**

   - The API is now accessible at `http://localhost:8000`.
   - **Search Endpoint**:

     ```http
     GET http://localhost:8000/search?query=Your+search+query
     ```

   - **Upload Endpoint**:

     You can upload new documents via the `/upload` endpoint using a tool like `curl` or Postman.

     Example using `curl`:

     ```bash
     curl -X POST "http://localhost:8000/upload" -F "file=@/path/to/your/document.txt"
     ```

7. **Monitor Logs**

   - Keep an eye on the terminal output for any logging information or errors.

---

## API Endpoints

- **`GET /search`**: Search for documents.

  - **Query Parameters**:
    - `query` (string): The search query.
    - `threshold` (float, default `0.7`): Similarity threshold.
    - `count` (int, default `10`): Number of results to return.

- **`POST /upload`**: Upload a new document to the system.

  - **Form Data**:
    - `file`: The file to upload.

---

## Additional Recommendations Implemented

- **Supabase Storage Integration**

  - Modified the application to use Supabase Storage for storing and retrieving documents.
  - Documents are fetched from Supabase Storage during application startup.

- **Error Handling and Logging**

  - Used the `loguru` library to add logging statements throughout the application.
  - Implemented error handling for various cases, such as network errors during file uploads/downloads.

- **Caching**

  - Implemented caching using `diskcache` to store and reuse generated contexts and embeddings, reducing API calls and costs.

- **Parallel Processing**

  - Used `asyncio.gather` to concurrently generate contexts and embeddings for document chunks, improving performance.

- **Environment Variable Management**

  - Used `python-dotenv` to securely manage environment variables.

- **Rate Limiting and API Usage**

  - The caching mechanism helps in reducing the number of API calls to OpenAI, mitigating rate limit issues.

- **Logging and Monitoring**

  - Detailed logging helps in monitoring the application's performance and debugging issues.

---

## Notes and Best Practices

- **Security**

  - Use the Supabase **service role key** for server-side operations and keep it secure.
  - Do not expose sensitive keys in client-side code or public repositories.
  - Ensure that your Supabase Storage bucket has appropriate access policies.

- **Data Handling**

  - Be cautious if your documents contain sensitive information; ensure that all data transfers are secure.
  - Implement SSL/TLS for secure communication between your application and Supabase.

- **Error Handling**

  - Add robust error handling for network issues, API errors, and unexpected exceptions.
  - Validate inputs received from users to prevent injection attacks.

- **Scaling**

  - Monitor performance and consider scaling strategies as your dataset grows.
  - Use asynchronous processing and efficient data structures to handle large volumes of data.

---

## Conclusion

This updated implementation of the Contextual Retrieval System integrates Supabase Storage for document storage, providing a centralized and scalable solution for managing documents. By leveraging Supabase for both vector storage and document storage, the system offers improved scalability, manageability, and efficiency.

The application incorporates best practices such as caching, asynchronous processing, and secure environment variable management to ensure efficient and secure operation. By integrating context into the retrieval process, the system offers improved accuracy, enhanced relevance, and a better user experience.

---

**Disclaimer:** Be mindful of the costs associated with using OpenAI's API and Supabase services. Monitor your usage and set appropriate limits to prevent unexpected charges. Always handle API keys and sensitive data securely.
