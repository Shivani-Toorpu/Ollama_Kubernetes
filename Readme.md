# Kubeflow LLM Local Setup

This setup provides a local Kubernetes environment with Ollama running Llama 3.2 and a Python backend service.

## Prerequisites

- Docker installed
- Kind installed
- kubectl installed

## Setup Steps

### 1. Create Kind Cluster
```bash
kind create cluster --config=kind-config.yaml
```

### 2. Deploy Ollama
```bash
kubectl apply -f ollama/ollama-deployment.yaml
```

Wait for Ollama to be ready:
```bash
kubectl wait --for=condition=available --timeout=300s deployment/ollama-deployment
```

### 3. Build and Deploy Backend

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

### 4. Verify Deployment

Check if all pods are running:
```bash
kubectl get pods
```

Check services:
```bash
kubectl get services
```

### 5. Test the Backend

The backend will be available at `http://localhost:8080`

Test health endpoint:
```bash
curl http://localhost:8080/health
```

Test query endpoint:
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Kubeflow?"}'
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /query` - Send questions to LLM
- `GET /models` - List available models

## Troubleshooting

### Check Ollama logs:
```bash
kubectl logs deployment/ollama-deployment
```

### Check backend logs:
```bash
kubectl logs deployment/kubeflow-backend
```

### Port forward for direct access:
```bash
# Ollama (internal use only)
kubectl port-forward service/ollama-service 11434:11434

# Backend
kubectl port-forward service/kubeflow-backend-service 8000:8000
```

## Notes

- Ollama is configured to run on CPU only (no GPU support in Kind)
- The Llama 3.2:3b model is automatically downloaded on first startup
- Backend service is exposed on NodePort 30080 (mapped to host port 8080)
- Ollama service is ClusterIP only (internal access)

## Cleanup

```bash
kind delete cluster --name kubeflow-llm-cluster
```