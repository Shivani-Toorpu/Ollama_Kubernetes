from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kubeflow LLM Backend", version="1.0.0")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama-service")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
MODEL_NAME = "llama3.2:3b"

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 1000
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int = 0

@app.get("/")
async def root():
    return {"message": "Kubeflow LLM Backend is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    try:
        # Check if Ollama is reachable
        response = requests.get(f"{OLLAMA_URL}/", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "ollama": "connected"}
        else:
            return {"status": "unhealthy", "ollama": "disconnected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.question[:100]}...")
        
        # Prepare the prompt for Kubeflow context
        prompt = f"""You are a helpful assistant for Kubeflow documentation and questions. 
Please provide accurate and helpful information about Kubeflow, Kubernetes, and MLOps.

Question: {request.question}

Answer:"""

        # Prepare request to Ollama
        ollama_request = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_k": 40,
                "top_p": 0.9
            }
        }
        
        # Send request to Ollama
        logger.info(f"Sending request to Ollama at {OLLAMA_URL}")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=ollama_request,
            timeout=120,  # 2 minutes timeout
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama request failed: {response.status_code}, {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Ollama request failed: {response.status_code}"
            )
        
        result = response.json()
        answer = result.get("response", "No response generated")
        
        logger.info("Successfully generated response")
        
        return QueryResponse(
            answer=answer,
            model=MODEL_NAME,
            tokens_used=len(answer.split())  # Rough token estimation
        )
        
    except requests.exceptions.Timeout:
        logger.error("Request to Ollama timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama service")
        raise HTTPException(status_code=503, detail="Could not connect to LLM service")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Could not fetch models")
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)