from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
from src.services.face.detector import RetinaFaceDetector, DetectedFace
from src.services.face.aligner import FaceAligner
from src.services.face.embedding import TensorRTEmbedder
from src.services.face.search import FAISSVectorSearch
from src.services.face.clusturing import FaceClusterer
import numpy as np
from typing import Tuple, Optional
import cv2


@dataclass
class FaceResult:
    """Complete face processing result."""
    face_id: str
    photo_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    thumbnail_path: str
    blur_score: float
    brightness_score: float


class FacePipeline:
    """
    End-to-end face processing pipeline.
    
    Workflow:
    1. Detect faces (RetinaFace)
    2. Align faces
    3. Generate embeddings (ArcFace + TensorRT)
    4. Store in vector DB (FAISS + PQ)
    5. Run clustering (HDBSCAN + Graph)
    """
    
    def __init__(
        self,
        detector: RetinaFaceDetector,
        aligner: FaceAligner,
        embedder: TensorRTEmbedder,
        search_engine: FAISSVectorSearch,
        clusterer: FaceClusterer
    ):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.search_engine = search_engine
        self.clusterer = clusterer
    
    def process_photo(
        self,
        photo_id: str,
        image: np.ndarray,
        save_crops: bool = True
    ) -> List[FaceResult]:
        """
        Process single photo: detect, align, embed.
        
        Args:
            photo_id: Photo UUID
            image: RGB image
            save_crops: Whether to save face crops
            
        Returns:
            List of FaceResult objects
        """
        results = []
        
        # 1. Detect faces
        detected_faces = self.detector.detect(image)
        
        if not detected_faces:
            return results
        
        # 2. Align faces
        aligned_faces = self.aligner.align_batch(image, detected_faces)
        
        # 3. Generate embeddings
        embeddings = self.embedder.embed_batch(aligned_faces)
        
        # 4. Process each face
        for i, (face, aligned, embedding) in enumerate(
            zip(detected_faces, aligned_faces, embeddings)
        ):
            face_id = str(uuid.uuid4())
            
            # Quality assessment
            blur_score = self._assess_blur(aligned)
            brightness_score = self._assess_brightness(aligned)
            
            # Save thumbnail (optional)
            thumbnail_path = None
            if save_crops:
                thumbnail_path = f"faces/{photo_id}/{face_id}.jpg"
                cv2.imwrite(thumbnail_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            
            results.append(FaceResult(
                face_id=face_id,
                photo_id=photo_id,
                bbox=face.bbox,
                confidence=face.confidence,
                embedding=embedding,
                thumbnail_path=thumbnail_path,
                blur_score=blur_score,
                brightness_score=brightness_score
            ))
        
        # 5. Add to search index
        face_ids = [r.face_id for r in results]
        emb_matrix = np.vstack([r.embedding for r in results])
        self.search_engine.add(emb_matrix, face_ids)
        
        return results
    
    def process_album(
        self,
        album_id: str,
        photos: List[Tuple[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Process entire album: detect, embed, cluster.
        
        Args:
            album_id: Album UUID
            photos: List of (photo_id, image) tuples
            
        Returns:
            Processing summary with clusters
        """
        all_results = []
        
        # Process all photos
        for photo_id, image in photos:
            results = self.process_photo(photo_id, image)
            all_results.extend(results)
        
        # Run clustering
        embeddings = np.vstack([r.embedding for r in all_results])
        face_ids = [r.face_id for r in all_results]
        photo_ids = [r.photo_id for r in all_results]
        
        cluster_map = self.clusterer.cluster(
            embeddings,
            face_ids,
            photo_ids
        )
        
        # Suggest merges
        merge_suggestions = self.clusterer.suggest_merges(
            cluster_map,
            embeddings,
            face_ids
        )
        
        return {
            'album_id': album_id,
            'total_faces': len(all_results),
            'total_photos': len(photos),
            'clusters': cluster_map,
            'num_persons': len(set(cluster_map.values())),
            'merge_suggestions': merge_suggestions,
            'results': all_results
        }
    
    def search_by_selfie(
        self,
        selfie_image: np.ndarray,
        album_id: Optional[str] = None,
        k: int = 50,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for matching faces using selfie.
        
        Args:
            selfie_image: Selfie image (RGB)
            album_id: Optional album filter
            k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of matching faces with scores
        """
        # Detect face in selfie
        faces = self.detector.detect(selfie_image, max_faces=1)
        
        if not faces:
            return []
        
        # Align and embed
        aligned = self.aligner.align(selfie_image, faces[0].landmarks)
        query_embedding = self.embedder.embed(aligned)
        
        # Search
        results = self.search_engine.search(
            query_embedding,
            k=k,
            threshold=threshold
        )
        
        return results
    
    @staticmethod
    def _assess_blur(image: np.ndarray) -> float:
        """Assess face blur using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 (higher = sharper)
        score = min(laplacian_var / 500.0, 1.0)
        return score
    
    @staticmethod
    def _assess_brightness(image: np.ndarray) -> float:
        """Assess face brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = gray.mean() / 255.0
        
        # Optimal range: 0.3 - 0.7
        if mean_brightness < 0.3:
            score = mean_brightness / 0.3
        elif mean_brightness > 0.7:
            score = (1.0 - mean_brightness) / 0.3
        else:
            score = 1.0
        
        return score