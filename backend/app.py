from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os, json, asyncio
import httpx
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSearcher:
    def __init__(self, milvus_host="milvus-service", milvus_port="19530"):
        self.client = None
        self.encoder = None
        self.repo_collections = {}
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.initialized = False
        
    def initialize(self):
        """Initialize the RAG components"""
        try:
            logger.info(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}")
            self.client = MilvusClient(uri=f"http://{self.milvus_host}:{self.milvus_port}")
            
            logger.info("Loading sentence transformer model...")
            self.encoder = SentenceTransformer("all-mpnet-base-v2")
            
            # Check for existing collections
            self.check_existing_collections()
            self.initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.initialized = False
    
    def check_existing_collections(self):
        """Check for existing repo collections"""
        try:
            collections = self.client.list_collections()
            for collection in collections:
                if collection.startswith("repo_"):
                    repo_name = collection.replace("repo_", "").replace("_", "-")
                    self.repo_collections[repo_name] = collection
                    logger.info(f"Found existing collection: {repo_name}")
        except Exception as e:
            logger.warning(f"Could not check collections: {e}")
    
    def search(self, query: str, limit: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Search across all repositories"""
        if not self.initialized:
            logger.warning("RAG system not initialized")
            return []
        
        try:
            # Get query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            all_results = []
            
            # Search across all collections
            for repo_name, collection_name in self.repo_collections.items():
                try:
                    results = self.client.search(
                        collection_name=collection_name,
                        data=[query_vector],
                        limit=limit,
                        output_fields=["repo_name", "filename", "filepath", "content"]
                    )
                    
                    # Format and filter results
                    for result in results[0]:
                        score = 1 - result['distance']
                        if score >= min_score:  # Only include relevant results
                            entity = result['entity']
                            all_results.append({
                                'repo_name': repo_name,
                                'filename': entity['filename'],
                                'filepath': entity['filepath'],
                                'content': entity['content'],
                                'score': score
                            })
                            
                except Exception as e:
                    logger.error(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def format_context(self, results: List[Dict]) -> str:
        """Format search results into context for the LLM"""
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Context {i}] File: {result['filepath']}\n"
                f"Content: {result['content']}\n"
                f"(Relevance: {result['score']:.2f})\n"
            )
        
        return "\n".join(context_parts)

# Initialize RAG searcher
rag_searcher = RAGSearcher(
    milvus_host=os.getenv('MILVUS_HOST', 'my-release-milvus'),
    milvus_port=os.getenv('MILVUS_PORT', '19530')
)

# Initialize in a separate thread to avoid blocking startup
def init_rag():
    rag_searcher.initialize()

threading.Thread(target=init_rag, daemon=True).start()

app = FastAPI()
OLLAMA_URL = f"http://{os.getenv('OLLAMA_HOST','ollama-service')}:{os.getenv('OLLAMA_PORT','11434')}"
MODEL = "llama3.2:3b"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_searcher.initialized,
        "collections": list(rag_searcher.repo_collections.keys()) if rag_searcher.initialized else []
    }

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            q = msg.get("question", "")
            max_tokens = msg.get("max_tokens", 500)  # Increased for context
            temp = msg.get("temperature", 0.7)
            use_rag = msg.get("use_rag", True)  # Allow disabling RAG

            if not q:
                await ws.send_text(json.dumps({"type":"error", "message":"Question required"}))
                continue

            await ws.send_text(json.dumps({"type":"start", "model": MODEL, "rag_enabled": use_rag}))

            # RAG Search
            context = ""
            search_results = []
            
            if use_rag and rag_searcher.initialized:
                try:
                    await ws.send_text(json.dumps({"type":"search", "message":"Searching knowledge base..."}))
                    search_results = rag_searcher.search(q, limit=3)
                    context = rag_searcher.format_context(search_results)
                    
                    if search_results:
                        await ws.send_text(json.dumps({
                            "type": "context_found",
                            "results_count": len(search_results),
                            "sources": [{"file": r['filepath'], "score": r['score']} for r in search_results]
                        }))
                    else:
                        await ws.send_text(json.dumps({"type":"search", "message":"No relevant context found, using general knowledge"}))
                        
                except Exception as e:
                    logger.error(f"RAG search failed: {e}")
                    await ws.send_text(json.dumps({"type":"warning", "message":"Search failed, using general knowledge"}))

            # Build prompt with context
            if context:
                prompt = (
                    "You are a helpful Kubeflow assistant. Use the provided context to answer questions accurately.\n"
                    "If the context doesn't contain relevant information, use your general knowledge but mention that.\n\n"
                    f"Context from Kubeflow documentation:\n{context}\n\n"
                    f"Question: {q}\n\n"
                    "Answer based on the context above:"
                )
            else:
                prompt = (
                    "You are a helpful Kubeflow assistant.\n"
                    f"Question: {q}\n\nAnswer:"
                )

            # Build options
            options = {"temperature": temp, "top_k": 40, "top_p": 0.9}
            if max_tokens and max_tokens > 0:
                options["num_predict"] = max_tokens

            req = {"model": MODEL, "prompt": prompt, "stream": True, "options": options}

            # Stream LLM response
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    async with client.stream(
                        "POST", f"{OLLAMA_URL}/api/generate", json=req
                    ) as r:
                        if r.status_code != 200:
                            await ws.send_text(
                                json.dumps({"type":"error", "message":f"LLM error {r.status_code}"})
                            )
                            continue

                        full, count = "", 0
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                                
                            text = chunk.get("response", "")
                            if not text:
                                continue

                            count += 1
                            full += text
                            await ws.send_text(json.dumps({
                                "type": "chunk", 
                                "content": text, 
                                "chunk": count
                            }))
                            
                            if ws.client_state.name != "CONNECTED":
                                return
                            await asyncio.sleep(0.01)

                            if chunk.get("done"):
                                await ws.send_text(json.dumps({
                                    "type": "complete", 
                                    "full_response": full, 
                                    "tokens": count,
                                    "context_used": len(search_results) > 0,
                                    "sources": [{"file": r['filepath'], "score": r['score']} for r in search_results] if search_results else []
                                }))
                                break
                                
                except httpx.TimeoutException:
                    await ws.send_text(json.dumps({"type":"error", "message":"Request timeout"}))
                except Exception as e:
                    logger.error(f"LLM request failed: {e}")
                    await ws.send_text(json.dumps({"type":"error", "message":f"LLM request failed: {str(e)}"}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        return
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.send_text(json.dumps({"type":"error", "message":f"Connection error: {str(e)}"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)