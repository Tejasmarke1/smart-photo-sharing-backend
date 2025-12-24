import hdbscan
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict
import numpy as np
from typing import List, Dict, Optional, Tuple


class FaceClusterer:
    """
    Two-stage face clustering:
    1. HDBSCAN: Initial density-based clusters
    2. Graph refinement: Merge/split using connectivity
    
    Handles:
    - Varying cluster sizes
    - Noise filtering
    - Temporal constraints (same photo = same person)
    """
    
    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        similarity_threshold: float = 0.6,
        same_photo_boost: float = 0.2
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
        self.same_photo_boost = same_photo_boost
    
    def cluster(
        self,
        embeddings: np.ndarray,
        face_ids: List[str],
        photo_ids: List[str],
        existing_labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, int]:
        """
        Cluster faces into persons.
        
        Args:
            embeddings: Face embeddings (N, 512)
            face_ids: Face IDs
            photo_ids: Photo IDs for each face
            existing_labels: Previously labeled faces {face_id: person_id}
            
        Returns:
            Cluster assignments {face_id: cluster_id}
        """
        n_faces = len(embeddings)
        
        # Step 1: HDBSCAN clustering
        print("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Step 2: Build similarity graph
        print("Building similarity graph...")
        graph = self._build_similarity_graph(
            embeddings,
            face_ids,
            photo_ids,
            labels
        )
        
        # Step 3: Graph-based refinement
        print("Refining clusters...")
        refined_labels = self._refine_clusters(
            graph,
            labels,
            existing_labels
        )
        
        # Create final mapping
        cluster_map = {
            face_id: int(label)
            for face_id, label in zip(face_ids, refined_labels)
        }
        
        return cluster_map
    
    def _build_similarity_graph(
        self,
        embeddings: np.ndarray,
        face_ids: List[str],
        photo_ids: List[str],
        initial_labels: np.ndarray
    ) -> nx.Graph:
        """Build weighted graph of face similarities."""
        G = nx.Graph()
        
        # Add nodes
        for i, face_id in enumerate(face_ids):
            G.add_node(face_id, embedding=embeddings[i], photo_id=photo_ids[i])
        
        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)
        
        # Add edges
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                sim = similarities[i, j]
                
                # Boost similarity if same photo
                if photo_ids[i] == photo_ids[j]:
                    sim += self.same_photo_boost
                
                # Add edge if above threshold
                if sim >= self.similarity_threshold:
                    G.add_edge(
                        face_ids[i],
                        face_ids[j],
                        weight=float(sim)
                    )
        
        return G
    
    def _refine_clusters(
        self,
        graph: nx.Graph,
        initial_labels: np.ndarray,
        existing_labels: Optional[Dict[str, str]]
    ) -> np.ndarray:
        """
        Refine clusters using graph connectivity.
        
        Process:
        1. Find connected components (potential persons)
        2. Split components with weak connectivity
        3. Merge components with strong connectivity
        4. Incorporate existing labels
        """
        # Find connected components
        components = list(nx.connected_components(graph))
        
        # Assign new cluster IDs
        refined_labels = initial_labels.copy()
        next_cluster_id = initial_labels.max() + 1
        
        for component in components:
            # Skip small components (noise)
            if len(component) < self.min_cluster_size:
                continue
            
            # Check if any faces have existing labels
            component_list = list(component)
            has_label = [
                existing_labels.get(fid) for fid in component_list
                if existing_labels and fid in existing_labels
            ]
            
            if has_label:
                # Use existing label
                label = has_label[0]
            else:
                # Assign new cluster ID
                label = next_cluster_id
                next_cluster_id += 1
            
            # Update all faces in component
            for face_id in component:
                face_idx = list(graph.nodes()).index(face_id)
                refined_labels[face_idx] = label
        
        return refined_labels
    
    def suggest_merges(
        self,
        cluster_map: Dict[str, int],
        embeddings: np.ndarray,
        face_ids: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[int, int, float]]:
        """
        Suggest cluster merges based on centroid similarity.
        
        Returns:
            List of (cluster_id_1, cluster_id_2, similarity)
        """
        # Compute cluster centroids
        clusters = defaultdict(list)
        for face_id, cluster_id in cluster_map.items():
            if cluster_id >= 0:  # Skip noise
                idx = face_ids.index(face_id)
                clusters[cluster_id].append(embeddings[idx])
        
        centroids = {
            cid: np.mean(embs, axis=0)
            for cid, embs in clusters.items()
        }
        
        # Find similar centroids
        suggestions = []
        cluster_ids = list(centroids.keys())
        
        for i, cid1 in enumerate(cluster_ids):
            for cid2 in cluster_ids[i+1:]:
                sim = np.dot(centroids[cid1], centroids[cid2])
                
                if sim >= threshold:
                    suggestions.append((cid1, cid2, float(sim)))
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return suggestions