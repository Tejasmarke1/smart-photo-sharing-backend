import faiss
from typing import Dict, Any,List
import pickle
import numpy as np


class FAISSVectorSearch:
    """
    FAISS-based vector search with Product Quantization.
    
    Index types:
    - Small (<100K): Flat (exact search)
    - Medium (100K-1M): IVF + PQ
    - Large (>1M): IVF + PQ + HNSW
    
    PQ compression: 512d -> 64 bytes (8x compression)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "IVF_PQ",
        nlist: int = 1024,  # Number of clusters for IVF
        m: int = 64,  # Number of subquantizers for PQ
        nbits: int = 8,  # Bits per subquantizer
        use_gpu: bool = True
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu
        
        self.index = None
        self.id_map = {}  # Maps FAISS index -> face_id
        
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index based on configuration."""
        if self.index_type == "Flat":
            # Exact search (baseline)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        elif self.index_type == "IVF_PQ":
            # IVF + Product Quantization
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                self.nlist,  # Number of clusters
                self.m,      # Subquantizers
                self.nbits   # Bits per subquantizer
            )
        
        elif self.index_type == "IVF_HNSW_PQ":
            # HNSW + IVF + PQ (best for large scale)
            quantizer = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                self.nlist,
                self.m,
                self.nbits
            )
        
        # Move to GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    def train(self, embeddings: np.ndarray):
        """
        Train index on embeddings.
        
        Required for IVF-based indices before adding vectors.
        
        Args:
            embeddings: Training embeddings (N, 512)
        """
        if not self.index.is_trained:
            print(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings.astype(np.float32))
            print("Training complete")
    
    def add(
        self,
        embeddings: np.ndarray,
        face_ids: List[str]
    ):
        """
        Add embeddings to index.
        
        Args:
            embeddings: Face embeddings (N, 512)
            face_ids: Corresponding face IDs
        """
        assert len(embeddings) == len(face_ids)
        
        # Train if needed
        if not self.index.is_trained:
            self.train(embeddings)
        
        # Get starting index
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embeddings.astype(np.float32))
        
        # Update ID mapping
        for i, face_id in enumerate(face_ids):
            self.id_map[start_idx + i] = face_id
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar faces.
        
        Args:
            query_embedding: Query embedding (512,)
            k: Number of results
            nprobe: Number of clusters to search (for IVF)
            threshold: Minimum similarity score
            
        Returns:
            List of matches with face_id and similarity score
        """
        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Reshape query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Filter and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            if score >= threshold:
                results.append({
                    'face_id': self.id_map.get(idx),
                    'similarity': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, path: str):
        """Save index and mappings."""
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{path}.faiss")
        else:
            faiss.write_index(self.index, f"{path}.faiss")
        
        # Save ID mapping
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.id_map, f)
    
    def load(self, path: str):
        """Load index and mappings."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Move to GPU if needed
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load ID mapping
        with open(f"{path}.pkl", 'rb') as f:
            self.id_map = pickle.load(f)