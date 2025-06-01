#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import tiktoken
import numpy as np
from tqdm import tqdm
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepoIndexer:
    def __init__(self, milvus_host="localhost", milvus_port="19530"):
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 512
        self.temp_dir = Path(tempfile.mkdtemp(prefix="repos_"))
        self.repo_collections = {}
        
        self.code_extensions = {
            '.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.h',
            '.rs', '.rb', '.php', '.yaml', '.yml', '.json', '.md', '.txt'
        }
        
        self.skip_dirs = {
            '.git', 'node_modules', 'vendor', '__pycache__', 'dist', 'build'
        }
    
    def check_existing_collections(self):
        """Check for existing repo collections"""
        try:
            collections = self.client.list_collections()
            for collection in collections:
                if collection.startswith("repo_"):
                    repo_name = collection.replace("repo_", "").replace("_", "-")
                    self.repo_collections[repo_name] = collection
                    print(f"✓ Found existing: {repo_name}")
        except Exception as e:
            logger.warning(f"Could not check collections: {e}")
    
    def clone_repo(self, repo_url: str) -> Path:
        """Clone repository"""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = self.temp_dir / repo_name
        
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        
        print(f"🔄 Cloning {repo_name}...")
        result = subprocess.run([
            "git", "clone", "--depth", "1", repo_url, str(repo_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Clone failed: {result.stderr}")
        
        # Remove .git folder
        git_dir = repo_path / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)
        
        return repo_path
    
    def get_files(self, repo_path: Path) -> List[Dict]:
        """Get all code files"""
        files = []
        
        print("📂 Scanning files...")
        for file_path in tqdm(list(repo_path.rglob("*")), desc="Scanning"):
            # Skip directories we don't want
            if any(skip in file_path.parts for skip in self.skip_dirs):
                continue
            
            # Only process code files under 1MB
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.code_extensions and
                file_path.stat().st_size < 1_000_000):
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if content.strip():
                        rel_path = file_path.relative_to(repo_path)
                        files.append({
                            'filename': file_path.name,
                            'filepath': str(rel_path).replace("\\", "/"),
                            'content': content
                        })
                except:
                    continue
        
        print(f"📄 Found {len(files)} files")
        return files
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size // 2):  # 50% overlap
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def process_files_to_chunks(self, files: List[Dict], repo_name: str) -> List[Dict]:
        """Convert files to chunks"""
        chunks = []
        
        print("✂️  Creating chunks...")
        for file_info in tqdm(files, desc="Chunking"):
            file_chunks = self.chunk_text(file_info['content'])
            
            for chunk_text in file_chunks:
                chunks.append({
                    'repo_name': repo_name,
                    'filename': file_info['filename'],
                    'filepath': file_info['filepath'],
                    'content': chunk_text
                })
        
        print(f"🧩 Created {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to chunks"""
        print("🔢 Creating embeddings...")
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            text = f"File: {chunk['filename']}\nPath: {chunk['filepath']}\n\n{chunk['content']}"
            texts.append(text)
        
        # Get embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            embeddings = self.encoder.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(embeddings)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['vector'] = embedding
        
        return chunks
    
    def create_collection(self, collection_name: str):
        """Create Milvus collection"""
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=768,  # all-mpnet-base-v2 dimension
            metric_type="COSINE"
        )
        print(f"📊 Created collection: {collection_name}")
    
    def insert_chunks(self, collection_name: str, chunks: List[Dict]):
        """Insert chunks into Milvus"""
        print("💾 Inserting into database...")
        
        # Prepare data for insertion
        data = []
        for i, chunk in enumerate(tqdm(chunks, desc="Preparing")):
            # Create unique int64 ID using hash
            id_string = f"{chunk['repo_name']}-{chunk['filepath']}-{i}"
            chunk_id = int(hashlib.md5(id_string.encode()).hexdigest()[:15], 16)
            
            data.append({
                "id": chunk_id,
                "vector": chunk['vector'].tolist(),
                "repo_name": chunk['repo_name'],
                "filename": chunk['filename'],
                "filepath": chunk['filepath'],
                "content": chunk['content']
            })
        
        # Insert in batches
        batch_size = 1000
        for i in tqdm(range(0, len(data), batch_size), desc="Inserting"):
            batch = data[i:i + batch_size]
            self.client.insert(collection_name=collection_name, data=batch)
        
        self.client.flush(collection_name)
        print(f"✅ Inserted {len(data)} chunks")
    
    def process_repo(self, repo_url: str, rebuild: bool = False) -> int:
        """Main processing pipeline"""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        collection_name = f"repo_{repo_name.replace('-', '_').lower()}"
        
        # Check if already exists
        if not rebuild and collection_name in self.repo_collections.values():
            stats = self.client.get_collection_stats(collection_name)
            count = stats.get('row_count', 0)
            print(f"⏭️  Skipping {repo_name} (already indexed: {count} chunks)")
            return count
        
        print(f"\n🚀 Processing: {repo_name}")
        
        try:
            # Step 1: Clone repo
            repo_path = self.clone_repo(repo_url)
            
            # Step 2: Get files
            files = self.get_files(repo_path)
            
            # Step 3: Create chunks
            chunks = self.process_files_to_chunks(files, repo_name)
            
            # Step 4: Add embeddings
            chunks = self.embed_chunks(chunks)
            
            # Step 5: Create collection
            self.create_collection(collection_name)
            self.repo_collections[repo_name] = collection_name
            
            # Step 6: Insert data
            self.insert_chunks(collection_name, chunks)
            
            # Cleanup
            shutil.rmtree(repo_path, ignore_errors=True)
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process {repo_name}: {e}")
            raise
    
    def search(self, repo_name: str, query: str, limit: int = 5) -> List[Dict]:
        """Search repository"""
        collection_name = self.repo_collections.get(repo_name)
        if not collection_name:
            print(f"❌ Repository '{repo_name}' not found")
            return []
        
        # Get query embedding
        query_vector = self.encoder.encode(query).tolist()
        
        # Search
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["repo_name", "filename", "filepath", "content"]
        )
        
        # Format results
        formatted = []
        for result in results[0]:
            entity = result['entity']
            formatted.append({
                'filename': entity['filename'],
                'filepath': entity['filepath'],
                'content': entity['content'],
                'score': 1 - result['distance']
            })
        
        return formatted
    
    def cleanup(self):
        """Clean up temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def main(rebuild=False):
    REPOS = [
        "https://github.com/kubeflow/website",
    ]
    
    indexer = RepoIndexer()
    
    try:
        # Check existing collections
        if not rebuild:
            indexer.check_existing_collections()
        
        # Process each repo
        for repo_url in REPOS:
            chunk_count = indexer.process_repo(repo_url, rebuild)
            print(f"✅ Completed with {chunk_count} chunks\n")
        
        # Test search
        print("=" * 50)
        print("🔍 TESTING SEARCH")
        print("=" * 50)
        
        query = "what is kubeflow trainer?"
        print(f"Query: '{query}'\n")
        
        for repo_name in indexer.repo_collections.keys():
            results = indexer.search(repo_name, query, limit=3)
            if results:
                print(f"Results from {repo_name}:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['filename']} (score: {result['score']:.3f})")
                    print(f"   {result['content']}...\n")
    
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise
    finally:
        indexer.cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Rebuild all indexes')
    args = parser.parse_args()
    
    main(rebuild=args.rebuild)