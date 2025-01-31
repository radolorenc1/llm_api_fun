from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="AI Completion Service")

# Add after app = FastAPI(title="AI Completion Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
)

# Models
class Question(BaseModel):
    content: str
    model: Optional[str] = "deepseek/deepseek-r1-distill-qwen-1.5b"

class Response(BaseModel):
    answer: str
    tokens_used: int
    timestamp: str

# Setup database
def init_db():
    conn = sqlite3.connect('usage.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys
                 (key text PRIMARY KEY, user_name text, created_date text)''')
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (api_key text, timestamp text, tokens_used integer)''')
    conn.commit()
    conn.close()

# Authentication
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    # In a real application, you would verify against your database
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required")
    # Add your API key verification logic here
    return api_key

# Routes
@app.post("/completion", response_model=Response)
async def get_completion(question: Question, api_key: str = Depends(verify_api_key)):
    try:
        completion = client.chat.completions.create(
            model=question.model,
            messages=[
                {
                    "role": "user",
                    "content": question.content
                }
            ]
        )
        
        response = Response(
            answer=completion.choices[0].message.content,
            tokens_used=completion.usage.total_tokens,
            timestamp=datetime.now().isoformat()
        )
        
        # Log usage
        conn = sqlite3.connect('usage.db')
        c = conn.cursor()
        c.execute("INSERT INTO usage_logs VALUES (?, ?, ?)",
                 (api_key, response.timestamp, response.tokens_used))
        conn.commit()
        conn.close()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage/{api_key}")
async def get_usage(api_key: str = Depends(verify_api_key)):
    conn = sqlite3.connect('usage.db')
    c = conn.cursor()
    c.execute("SELECT SUM(tokens_used) FROM usage_logs WHERE api_key = ?", (api_key,))
    total_tokens = c.fetchone()[0] or 0
    conn.close()
    return {"total_tokens_used": total_tokens}

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000) 