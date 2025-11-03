# Trade War Economic Assistant

A Python-based economic policy evaluation system that uses RAG (Retrieval Augmented Generation) to analyze trade policies and predict economic impacts.

## Setup Instructions

### 1. Create a Python Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here

# MySQL Database Configuration
DB_HOST=your_db_host
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DB_PORT=3306

# CORS Configuration
ALLOW_ORIGINS=http://localhost:3000
```

### 4. Initialize Embeddings

Load documents and Wikipedia data into Pinecone:

```bash
# Load documents from the docs folder
python init_embeddings/load_docs.py

# Load Wikipedia data for countries
python init_embeddings/load_wiki.py
```

### 5. Run the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8080`

## Project Structure

- `api.py` - FastAPI server and endpoints
- `rag_agent.py` - RAG agent for policy evaluation
- `db_utils.py` - Database utilities
- `init_embeddings/` - Embedding initialization scripts
  - `load_docs.py` - Load PDF and MHTML documents
  - `load_wiki.py` - Load Wikipedia data
  - `search.py` - Search and test embeddings
  - `docs/` - Document storage

## API Endpoints

### POST `/api/eval-trade-policy`

Evaluate a trade policy strategy.

**Request Body:**
```json
{
  "strategy": "Impose 40% tariffs on Chinese imports",
  "country": "USA",
  "opposing_country": "China",
  "session_id": "unique_session_id"
}
```

**Response:**
```json
{
  "strategy": "Impose 40% tariffs on Chinese imports",
  "country": "USA",
  "GDP_change": -0.5,
  "Political_boost": 2,
  "Trade_balance_shift": 0.5,
  "confidence": "Medium",
  "rational": "...",
  "source_refs": []
}
```

## Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```
