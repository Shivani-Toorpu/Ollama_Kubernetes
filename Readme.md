# Kubeflow LLM Local Setup with RAG

This setup provides a local Kubernetes environment with Ollama running Llama 3.2, a Python backend service with RAG (Retrieval-Augmented Generation) capabilities powered by Milvus vector database.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────│  Backend API    │────│    Ollama LLM   │
│                 │    │  (FastAPI +     │    │  (Llama 3.2)    │
│                 │    │   WebSocket)    │    │                 │
└─────────────────┘    └─────────┬───────┘    └─────────────────┘
                                 │
                                 │
                       ┌─────────▼───────┐
                       │  Milvus Vector  │
                       │    Database     │
                       │ (Semantic Search)│
                       └─────────────────┘
```

## Prerequisites

- Docker installed and running
- Kind installed (`go install sigs.k8s.io/kind@latest`)
- kubectl installed
- Python 3.11+ (for local development/indexing)
- Git (for cloning repositories to index)

## Setup Steps

### 1. Create Kind Cluster
```bash
kind create cluster --config=kind-config.yaml --name kubeflow-llm-cluster
```

### 2. Deploy Milvus Vector Database
```bash
kubectl apply -f milvus/milvus-deployment.yaml
```

Wait for Milvus to be ready:
```bash
kubectl wait --for=condition=available --timeout=300s deployment/milvus-deployment
```

### 3. Deploy Ollama
```bash
kubectl apply -f ollama/ollama-deployment.yaml
```

Wait for Ollama to be ready:
```bash
kubectl wait --for=condition=available --timeout=300s deployment/ollama-deployment
```

### 4. Index Kubeflow Documentation (RAG Setup)

**Option A: Index from outside the cluster (Recommended)**
```bash
# Install indexing dependencies
pip install pymilvus sentence-transformers tiktoken tqdm

# Port forward Milvus for indexing
kubectl port-forward service/milvus-service 19530:19530 &

# Run the indexer script
python indexer/repo_indexer.py

# Stop port forwarding
pkill -f "port-forward.*milvus"
```

**Option B: Run indexer as a Kubernetes Job**
```bash
# Build indexer image
cd indexer
docker build -t kubeflow-indexer:latest .
kind load docker-image kubeflow-indexer:latest --name kubeflow-llm-cluster

# Run indexing job
kubectl apply -f indexer/indexer-job.yaml

# Monitor indexing progress
kubectl logs -f job/kubeflow-indexer
```

### 5. Build and Deploy Backend

Build the Docker image:
```bash
cd backend
docker build -t kubeflow-backend:latest .
```

Load image into Kind cluster:
```bash
kind load docker-image kubeflow-backend:latest --name kubeflow-llm-cluster
```

Deploy the backend:
```bash
kubectl apply -f backend/backend-deployment.yaml
```

### 6. Verify Deployment

Check if all pods are running:
```bash
kubectl get pods
```

Expected output:
```
NAME                                  READY   STATUS    RESTARTS   AGE
kubeflow-backend-xxx                  1/1     Running   0          2m
milvus-deployment-xxx                 1/1     Running   0          5m
ollama-deployment-xxx                 1/1     Running   0          4m
```

Check services:
```bash
kubectl get services
```

### 7. Test the System

#### Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "collections": ["website"]
}
```

#### WebSocket Test (using wscat)
```bash
# Install wscat if needed
npm install -g wscat

# Connect and test
wscat -c ws://localhost:8080/ws

# Send a message
{"question": "What is Kubeflow Pipelines?", "use_rag": true, "max_tokens": 300}
```

## API Reference

### WebSocket Endpoint: `/ws`

**Message Format:**
```json
{
  "question": "Your question here",
  "use_rag": true,
  "max_tokens": 500,
  "temperature": 0.7
}
```

**Response Types:**
- `start` - Query processing started
- `search` - RAG search status
- `context_found` - Relevant context found
- `chunk` - Streaming response chunk
- `complete` - Response completed with metadata
- `error` - Error occurred

### REST Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check with RAG status

## Configuration

### Environment Variables

**Backend Service:**
- `OLLAMA_HOST` - Ollama service hostname (default: `ollama-service`)
- `OLLAMA_PORT` - Ollama service port (default: `11434`)
- `MILVUS_HOST` - Milvus service hostname (default: `milvus-service`)
- `MILVUS_PORT` - Milvus service port (default: `19530`)

**Indexer:**
- `MILVUS_URI` - Full Milvus connection URI
- `REBUILD_INDEX` - Force rebuild existing indexes (`true`/`false`)

### Customizing Repositories to Index

Edit `indexer/repo_indexer.py` and modify the `REPOS` list:
```python
REPOS = [
    "https://github.com/kubeflow/website",
    "https://github.com/kubeflow/pipelines",
    "https://github.com/kubeflow/training-operator",
    # Add more repositories here
]
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -o wide
```

### View Logs
```bash
# Backend logs
kubectl logs deployment/kubeflow-backend -f

# Ollama logs
kubectl logs deployment/ollama-deployment -f

# Milvus logs
kubectl logs deployment/milvus-deployment -f
```

### Test Individual Components

**Test Ollama directly:**
```bash
kubectl port-forward service/ollama-service 11434:11434 &
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "What is Kubernetes?",
  "stream": false
}'
```

**Test Milvus directly:**
```bash
kubectl port-forward service/milvus-service 19530:19530 &
python -c "
from pymilvus import MilvusClient
client = MilvusClient('http://localhost:19530')
print('Collections:', client.list_collections())
"
```

### Common Issues

**1. Pods stuck in Pending state:**
```bash
kubectl describe pod <pod-name>
```
Usually indicates resource constraints or image pull issues.

**2. RAG not finding relevant results:**
- Check if indexing completed successfully
- Verify collections exist: `curl http://localhost:8080/health`
- Try different search queries or adjust `min_score` in the code

**3. Ollama model download slow:**
The Llama 3.2:3b model (~2GB) downloads on first startup. Monitor with:
```bash
kubectl logs deployment/ollama-deployment -f
```

**4. WebSocket connection issues:**
- Ensure backend service is running
- Check NodePort mapping: `kubectl get svc kubeflow-backend-service`
- Test with different WebSocket clients

### Performance Tuning

**For better performance:**
1. Increase backend replicas:
   ```bash
   kubectl scale deployment kubeflow-backend --replicas=3
   ```

2. Adjust chunk size and embedding batch size in indexer
3. Use GPU-enabled nodes for Ollama (requires GPU operator)

## Development

### Local Development
```bash
# Run backend locally (requires port-forwarding)
kubectl port-forward service/milvus-service 19530:19530 &
kubectl port-forward service/ollama-service 11434:11434 &
cd backend
pip install -r requirements.txt
python app.py
```

### Adding New Models
Edit `ollama/ollama-deployment.yaml` and modify the init container:
```yaml
- name: model-puller
  image: curlimages/curl:latest
  command: ["/bin/sh"]
  args:
    - -c
    - |
      curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2:3b"}'
      curl -X POST http://localhost:11434/api/pull -d '{"name": "codellama:7b"}'
```

### Extending RAG
- Add new file types in `code_extensions`
- Implement custom chunking strategies
- Add metadata filtering
- Integrate multiple vector databases

## Monitoring

### Resource Usage
```bash
kubectl top pods
kubectl top nodes
```

### Metrics Collection
For production deployments, consider adding:
- Prometheus for metrics
- Grafana for visualization
- Jaeger for tracing

## Security Considerations

- Services use ClusterIP by default (internal only)
- Consider adding authentication for production
- Network policies for pod-to-pod communication
- Resource limits and quotas

## Scaling

### Horizontal Scaling
```bash
# Scale backend
kubectl scale deployment kubeflow-backend --replicas=3

# Scale Ollama (if multiple models needed)
kubectl scale deployment ollama-deployment --replicas=2
```

### Vertical Scaling
Edit deployment files to adjust resource requests/limits:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi" 
    cpu: "4"
```

## Cleanup

Remove the entire setup:
```bash
kind delete cluster --name kubeflow-llm-cluster
```

Remove specific deployments:
```bash
kubectl delete -f backend/backend-deployment.yaml
kubectl delete -f ollama/ollama-deployment.yaml
kubectl delete -f milvus/milvus-deployment.yaml
```

## Next Steps

1. **Add Authentication**: Implement JWT or OAuth2
2. **Web UI**: Build a React/Vue frontend
3. **Multiple Models**: Support model switching
4. **Batch Processing**: Add batch query endpoints
5. **Caching**: Implement Redis for response caching
6. **Monitoring**: Add comprehensive observability
7. **CI/CD**: Automate deployments with GitOps

## Contributing

1. Fork the repository
2. Create feature branches
3. Test changes locally
4. Submit pull requests with detailed descriptions

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

For issues or questions, please open an issue in the repository or check the troubleshooting section above.