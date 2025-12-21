#!/usr/bin/env python3
"""
Local Face Recognition Testing Script
======================================

Test the face recognition pipeline on your local images without any server setup.

Usage:
    python test_local_images.py /path/to/your/images/
    python test_local_images.py /path/to/single_image.jpg
    python test_local_images.py /path/to/images/ --show-faces
    python test_local_images.py /path/to/images/ --cluster --output results/

Requirements:
    pip install opencv-python pillow numpy insightface
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import sys

# Try to import required packages
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace not available. Face embeddings will be disabled.")
    print("   Install with: pip install insightface")

try:
    from sklearn.cluster import DBSCAN
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("⚠️  scikit-learn not available. Clustering will be disabled.")
    print("   Install with: pip install scikit-learn")


class LocalFaceDetector:
    """Simple face detector using OpenCV's Haar Cascade."""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("✓ Face detector initialized (Haar Cascade)")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'confidence': 0.99
            })
        
        return results


class LocalFaceEmbedder:
    """Face embedder using InsightFace."""
    
    def __init__(self):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not available")
        
        print("⏳ Initializing InsightFace (this may take a moment)...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ Face embedder initialized (InsightFace/ArcFace)")
    
    def embed(self, face_image: np.ndarray) -> np.ndarray:
        """Generate embedding for face crop."""
        # InsightFace expects BGR
        faces = self.app.get(face_image)
        
        if len(faces) == 0:
            return np.zeros(512, dtype=np.float32)
        
        return faces[0].normed_embedding


class FacePipeline:
    """Complete face processing pipeline for local testing."""
    
    def __init__(self, use_embeddings: bool = True):
        self.detector = LocalFaceDetector()
        self.embedder = None
        
        if use_embeddings and INSIGHTFACE_AVAILABLE:
            try:
                self.embedder = LocalFaceEmbedder()
            except Exception as e:
                print(f"⚠️  Could not initialize embedder: {e}")
                print("   Continuing without embeddings...")
    
    def process_image(
        self,
        image_path: Path,
        save_faces: bool = False,
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Process single image: detect faces and generate embeddings.
        
        Returns:
            Dict with image info, detected faces, and embeddings
        """
        print(f"\n📸 Processing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ Could not read image")
            return None
        
        height, width = image.shape[:2]
        print(f"   Image size: {width}x{height}")
        
        # Detect faces
        detected_faces = self.detector.detect(image)
        print(f"   Found {len(detected_faces)} face(s)")
        
        if len(detected_faces) == 0:
            return {
                'path': str(image_path),
                'width': width,
                'height': height,
                'faces': []
            }
        
        # Process each face
        faces_data = []
        for i, face in enumerate(detected_faces):
            x, y, w, h = face['bbox']
            print(f"   Face {i+1}: bbox=({x}, {y}, {w}, {h}), conf={face['confidence']:.2f}")
            
            # Extract face crop
            face_crop = image[y:y+h, x:x+w]
            
            # Generate embedding if available
            embedding = None
            if self.embedder:
                face_resized = cv2.resize(face_crop, (112, 112))
                embedding = self.embedder.embed(face_resized)
                print(f"   → Generated embedding (dim={len(embedding)})")
            
            # Save face crop if requested
            crop_path = None
            if save_faces and output_dir:
                crop_path = output_dir / f"{image_path.stem}_face_{i+1}.jpg"
                cv2.imwrite(str(crop_path), face_crop)
                print(f"   → Saved crop: {crop_path.name}")
            
            faces_data.append({
                'face_id': i + 1,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'embedding': embedding.tolist() if embedding is not None else None,
                'crop_path': str(crop_path) if crop_path else None
            })
        
        return {
            'path': str(image_path),
            'width': width,
            'height': height,
            'faces': faces_data
        }
    
    def process_directory(
        self,
        directory: Path,
        save_faces: bool = False,
        output_dir: Path = None
    ) -> List[Dict[str, Any]]:
        """Process all images in directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in directory.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No images found in {directory}")
            return []
        
        print(f"\n🔍 Found {len(image_files)} image(s) in {directory}")
        
        results = []
        for image_path in sorted(image_files):
            result = self.process_image(image_path, save_faces, output_dir)
            if result:
                results.append(result)
        
        return results


def cluster_faces(results: List[Dict[str, Any]], threshold: float = 0.6) -> Dict[int, List[Dict]]:
    """
    Cluster faces across all images using DBSCAN.
    
    Returns:
        Dict mapping cluster_id -> list of face records
    """
    if not CLUSTERING_AVAILABLE:
        print("❌ Clustering not available (scikit-learn not installed)")
        return {}
    
    # Collect all embeddings
    all_faces = []
    for result in results:
        for face in result['faces']:
            if face['embedding']:
                all_faces.append({
                    'image_path': result['path'],
                    'face_id': face['face_id'],
                    'embedding': face['embedding'],
                    'bbox': face['bbox']
                })
    
    if len(all_faces) < 2:
        print("⚠️  Not enough faces for clustering")
        return {}
    
    print(f"\n🔬 Clustering {len(all_faces)} faces...")
    
    # Convert to numpy array
    embeddings = np.array([f['embedding'] for f in all_faces])
    
    # Use DBSCAN for clustering
    # eps = 1 - threshold (distance threshold)
    clusterer = DBSCAN(eps=1-threshold, min_samples=2, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
    
    # Group by cluster
    clusters = {}
    noise_count = 0
    
    for face, label in zip(all_faces, labels):
        if label == -1:
            noise_count += 1
            continue
        
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append(face)
    
    print(f"   Found {len(clusters)} person(s)")
    print(f"   Noise faces: {noise_count}")
    
    for cluster_id, faces in clusters.items():
        print(f"   Person {cluster_id + 1}: {len(faces)} face(s)")
    
    return clusters


def draw_results(
    image_path: Path,
    faces: List[Dict[str, Any]],
    output_dir: Path,
    cluster_id: int = None
) -> Path:
    """Draw bounding boxes on image and save."""
    image = cv2.imread(str(image_path))
    
    for face in faces:
        x, y, w, h = face['bbox']
        
        # Different colors for different clusters
        if cluster_id is not None:
            colors = [
                (0, 255, 0),   # Green
                (255, 0, 0),   # Blue
                (0, 0, 255),   # Red
                (255, 255, 0), # Cyan
                (255, 0, 255), # Magenta
                (0, 255, 255), # Yellow
            ]
            color = colors[cluster_id % len(colors)]
        else:
            color = (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"Person {cluster_id + 1}" if cluster_id is not None else f"Face {face['face_id']}"
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    # Save
    output_path = output_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_path), image)
    
    return output_path


def generate_report(
    results: List[Dict[str, Any]],
    clusters: Dict[int, List[Dict]],
    output_dir: Path
):
    """Generate HTML report with results."""
    total_images = len(results)
    total_faces = sum(len(r['faces']) for r in results)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        .image-card img {{ max-width: 100%; height: auto; }}
        .cluster {{ background: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🎭 Face Recognition Results</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Images:</strong> {total_images}</p>
        <p><strong>Total Faces:</strong> {total_faces}</p>
        <p><strong>Persons Found:</strong> {len(clusters)}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    # Clusters section
    if clusters:
        html += "<h2>👥 Detected Persons</h2>"
        for cluster_id, faces in clusters.items():
            html += f"""
            <div class="cluster">
                <h3>Person {cluster_id + 1} ({len(faces)} faces)</h3>
                <div class="image-grid">
"""
            for face in faces[:6]:  # Show max 6 faces per person
                img_name = Path(face['image_path']).name
                html += f"""
                <div class="image-card">
                    <img src="annotated_{img_name}" alt="Face">
                    <p>{img_name}</p>
                </div>
"""
            html += """
                </div>
            </div>
"""
    
    # All images section
    html += "<h2>📸 All Images</h2><div class='image-grid'>"
    
    for result in results:
        img_name = Path(result['path']).name
        html += f"""
        <div class="image-card">
            <img src="annotated_{img_name}" alt="{img_name}">
            <p><strong>{img_name}</strong></p>
            <p>Faces: {len(result['faces'])}</p>
            <p>Size: {result['width']}x{result['height']}</p>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Test face recognition pipeline on local images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python test_local_images.py photo.jpg
  
  # Process all images in directory
  python test_local_images.py /path/to/photos/
  
  # Save face crops
  python test_local_images.py /path/to/photos/ --save-faces
  
  # Cluster faces and generate report
  python test_local_images.py /path/to/photos/ --cluster --report
  
  # Specify output directory
  python test_local_images.py /path/to/photos/ --output results/
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to image file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='face_results',
        help='Output directory for results (default: face_results)'
    )
    
    parser.add_argument(
        '--save-faces',
        action='store_true',
        help='Save individual face crops'
    )
    
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='Cluster faces to identify same person across images'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML report'
    )
    
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip embedding generation (faster but no clustering)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Similarity threshold for clustering (default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize pipeline
    use_embeddings = not args.no_embeddings
    pipeline = FacePipeline(use_embeddings=use_embeddings)
    
    # Process images
    if input_path.is_file():
        results = [pipeline.process_image(input_path, args.save_faces, output_dir)]
        results = [r for r in results if r]  # Filter None
    else:
        results = pipeline.process_directory(input_path, args.save_faces, output_dir)
    
    if not results:
        print("\n❌ No results to process")
        sys.exit(1)
    
    # Save results to JSON
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Clustering
    clusters = {}
    if args.cluster and use_embeddings:
        clusters = cluster_faces(results, args.threshold)
        
        if clusters:
            # Save clusters to JSON
            clusters_json = output_dir / 'clusters.json'
            clusters_serializable = {
                int(k): [
                    {**face, 'embedding': None}  # Remove embeddings for JSON
                    for face in v
                ]
                for k, v in clusters.items()
            }
            with open(clusters_json, 'w') as f:
                json.dump(clusters_serializable, f, indent=2)
            print(f"💾 Clusters saved to: {clusters_json}")
    
    # Draw results on images
    print(f"\n🎨 Generating annotated images...")
    for result in results:
        image_path = Path(result['path'])
        
        if clusters and result['faces']:
            # Find cluster for each face
            for face in result['faces']:
                for cluster_id, cluster_faces in clusters.items():
                    if any(
                        f['image_path'] == result['path'] and 
                        f['face_id'] == face['face_id']
                        for f in cluster_faces
                    ):
                        face['cluster_id'] = cluster_id
                        break
        
        annotated = draw_results(
            image_path,
            result['faces'],
            output_dir,
            cluster_id=result['faces'][0].get('cluster_id') if result['faces'] else None
        )
        print(f"   ✓ {annotated.name}")
    
    # Generate report
    if args.report:
        print(f"\n📊 Generating HTML report...")
        report_path = generate_report(results, clusters, output_dir)
        print(f"   ✓ Report: {report_path}")
        print(f"\n🌐 Open in browser: file://{report_path.absolute()}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"✅ Processing complete!")
    print(f"="*60)
    print(f"📊 Statistics:")
    print(f"   • Images processed: {len(results)}")
    print(f"   • Total faces found: {sum(len(r['faces']) for r in results)}")
    if clusters:
        print(f"   • Persons identified: {len(clusters)}")
    print(f"\n📁 Results in: {output_dir.absolute()}")
    
    if args.report:
        print(f"\n👉 View report: file://{report_path.absolute()}")


if __name__ == "__main__":
    main()