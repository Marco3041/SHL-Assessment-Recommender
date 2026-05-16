# SHL Assessment Recommender

A conversational AI agent for recommending SHL Individual Test Assessments.

## Setup

### 1. Get a Gemini API Key (Free)
1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. Copy the key

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Scraper (already done - catalog.json included)
```bash
python scraper.py
```

### 5. Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /health
→ {"status": "ok"}
```

### Chat
```
POST /chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}

→ {
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    ...
  ],
  "end_of_conversation": false
}
```

## Architecture

- **FastAPI** - REST API framework
- **Google Gemini 1.5 Flash** - Free-tier LLM for conversation
- **Sentence Transformers** (`all-MiniLM-L6-v2`) - Text embedding for semantic search
- **FAISS** - Vector index for fast similarity search (RAG retrieval)
- **SHL Catalog** - 377 Individual Test Solutions scraped from shl.com

## How It Works

1. User sends conversation history to `/chat`
2. Agent extracts search query from last 3 user messages
3. FAISS retrieves top-15 relevant catalog items
4. Gemini LLM generates response with catalog context injected
5. Response is validated (URLs must be from catalog)
6. Structured JSON response returned

## Deployment (Render.com)

1. Push to GitHub
2. Connect repo to Render.com
3. Set `GEMINI_API_KEY` environment variable
4. Deploy (render.yaml is pre-configured)

## Running Tests
```bash
python test_agent.py
```
