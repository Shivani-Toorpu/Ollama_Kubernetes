#!/usr/bin/env python3
from pymilvus import MilvusClient

class MilvusCleaner:
    def __init__(self, host="localhost", port="19530"):
        self.client = MilvusClient(uri=f"http://{host}:{port}")
    
    def clean_all(self):
        """Drop all collections"""
        collections = self.client.list_collections()
        
        if not collections:
            print("No collections found")
            return
        
        print(f"Found {len(collections)} collections")
        
        for collection in collections:
            try:
                self.client.drop_collection(collection)
                print(f"✅ Dropped: {collection}")
            except Exception as e:
                print(f"❌ Failed to drop {collection}: {e}")
        
        print("🧹 Cleanup complete")

if __name__ == "__main__":
    cleaner = MilvusCleaner()
    cleaner.clean_all()