#!/usr/bin/env python3
"""
Local CPU Face Recognition Testing
===================================

Test your face recognition pipeline on local images using CPU.

Usage:
    python test_cpu_local.py /path/to/images/
    python test_cpu_local.py photo.jpg
    python test_cpu_local.py /path/to/images/ --save-results
    python test_cpu_local.py /path/to/images/ --cluster
    
Requirements:
    pip install opencv-python pillow numpy insightface scikit-learn
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from datetime import datetime
import sys

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'    


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


# =============================================================================
# Simple CPU-Optimized Face Pipeline
# =============================================================================

class SimpleCPUFacePipeline:
    """
    Simplified face recognition pipeline optimized for CPU.
    
    Uses InsightFace with ONNX Runtime (CPU backend).
    """
    
    def __init__(self):
        """Initialize pipeline with CPU-optimized settings."""
        print_header("Initializing Face Recognition Pipeline (CPU)")
        
        try:
            from insightface.app import FaceAnalysis
            
            print_info("Loading InsightFace model (this may take 30-60 seconds)...")
            start = time.time()
            
            # Initialize with CPU settings
            self.app = FaceAnalysis(
                name='buffalo_l',  # Production-quality model
                providers=['CPUExecutionProvider']  # CPU only
            )
            
            # Prepare with CPU-optimized settings
            self.app.prepare(
                ctx_id=-1,  # CPU context
                det_size=(480, 480),  # Detection size (balanced for CPU)
                det_thresh=0.5  # Detection threshold
            )
            
            elapsed = time.time() - start
            
            print_success(f"Pipeline initialized in {elapsed:.2f}s")
            print_info(f"Backend: ONNX Runtime (CPU)")
            print_info(f"Model: buffalo_l (99.83% accuracy)")
            print_info(f"Detection size: 480x480")
            
            self.initialized = True
            
        except ImportError as e:
            print_error("InsightFace not installed!")
            print_info("Install with: pip install insightface onnxruntime opencv-python")
            self.initialized = False
            raise
    
    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process single image: detect faces and generate embeddings.
        
        Args:
            image: RGB or BGR image
            
        Returns:
            List of face dictionaries
        """
        if not self.initialized:
            return []
        
        start = time.time()
        
        # Detect faces (InsightFace expects BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR if needed
            if image[0, 0, 0] > image[0, 0, 2]:  # Simple heuristic
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
        else:
            image_bgr = image
        
        faces = self.app.get(image_bgr)
        
        elapsed = time.time() - start
        
        results = []
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            result = {
                'face_id': i + 1,
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # x, y, w, h
                'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],  # x1, y1, x2, y2
                'confidence': float(face.det_score),
                'embedding': face.normed_embedding.tolist(),
                'landmarks': face.kps.tolist(),
                'age': int(face.age) if hasattr(face, 'age') else None,
                'gender': 'M' if (hasattr(face, 'gender') and face.gender == 1) else 'F',
            }
            
            results.append(result)
        
        return results, elapsed
    
    def compare_faces(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two face embeddings.
        
        Returns similarity score (0-1, higher = more similar).
        """
        return float(np.dot(emb1, emb2))


# =============================================================================
# Image Processing
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """Load image from file."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def save_annotated_image(
    image: np.ndarray,
    faces: List[Dict],
    output_path: Path
):
    """Draw face boxes and save annotated image."""
    img_annotated = image.copy()
    
    for face in faces:
        x1, y1, x2, y2 = face['bbox_xyxy']
        confidence = face['confidence']
        face_id = face['face_id']
        
        # Draw rectangle
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Face {face_id} ({confidence:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            img_annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # Draw landmarks
        if 'landmarks' in face:
            for point in face['landmarks']:
                cv2.circle(img_annotated, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    
    cv2.imwrite(str(output_path), img_annotated)


def save_face_crop(
    image: np.ndarray,
    bbox: List[int],
    output_path: Path
):
    """Save cropped face."""
    x, y, w, h = bbox
    
    # Add padding
    pad = int(max(w, h) * 0.2)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), crop)


# =============================================================================
# Clustering
# =============================================================================

def cluster_faces(all_faces: List[Dict], threshold: float = 0.6) -> Dict[int, List[Dict]]:
    """
    Cluster faces across all images.
    
    Args:
        all_faces: List of all detected faces
        threshold: Similarity threshold for clustering
        
    Returns:
        Dictionary mapping cluster_id to list of faces
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print_warning("scikit-learn not installed, skipping clustering")
        print_info("Install with: pip install scikit-learn")
        return {}
    
    if len(all_faces) < 2:
        print_warning("Need at least 2 faces for clustering")
        return {}
    
    print_info(f"Clustering {len(all_faces)} faces...")
    
    # Extract embeddings
    embeddings = np.array([face['embedding'] for face in all_faces])
    
    # Use DBSCAN with cosine distance
    clusterer = DBSCAN(
        eps=1 - threshold,  # Convert similarity to distance
        min_samples=1,
        metric='cosine'
    )
    
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
    
    print_success(f"Found {len(clusters)} person(s)")
    if noise_count > 0:
        print_info(f"Noise faces (not clustered): {noise_count}")
    
    for cluster_id, faces in clusters.items():
        print_info(f"  Person {cluster_id + 1}: {len(faces)} face(s)")
    
    return clusters


# =============================================================================
# Results & Reporting
# =============================================================================

def generate_summary(results: List[Dict], total_time: float) -> Dict:
    """Generate processing summary."""
    total_images = len(results)
    total_faces = sum(len(r['faces']) for r in results)
    avg_faces_per_image = total_faces / total_images if total_images > 0 else 0
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    
    return {
        'total_images': total_images,
        'total_faces': total_faces,
        'avg_faces_per_image': avg_faces_per_image,
        'total_processing_time': total_time,
        'avg_time_per_image': avg_time_per_image,
        'timestamp': datetime.now().isoformat()
    }


def print_summary(summary: Dict):
    """Print processing summary."""
    print_header("Processing Summary")
    
    print(f"Total Images: {Colors.BOLD}{summary['total_images']}{Colors.ENDC}")
    print(f"Total Faces: {Colors.BOLD}{summary['total_faces']}{Colors.ENDC}")
    print(f"Avg Faces/Image: {Colors.BOLD}{summary['avg_faces_per_image']:.1f}{Colors.ENDC}")
    print(f"Total Time: {Colors.BOLD}{summary['total_processing_time']:.2f}s{Colors.ENDC}")
    print(f"Avg Time/Image: {Colors.BOLD}{summary['avg_time_per_image']:.2f}s{Colors.ENDC}")


# =============================================================================
# Main Processing
# =============================================================================

def process_single_image(
    pipeline: SimpleCPUFacePipeline,
    image_path: Path,
    output_dir: Path,
    save_crops: bool = False
) -> Dict:
    """Process a single image."""
    print(f"\n📸 Processing: {Colors.BOLD}{image_path.name}{Colors.ENDC}")
    
    # Load image
    image = load_image(image_path)
    height, width = image.shape[:2]
    print_info(f"Image size: {width}x{height}")
    
    # Process
    faces, elapsed = pipeline.process_image(image)
    
    print_success(f"Found {len(faces)} face(s) in {elapsed*1000:.0f}ms")
    
    # Print face details
    for face in faces:
        print(f"  Face {face['face_id']}: "
              f"bbox={face['bbox']}, "
              f"conf={face['confidence']:.3f}, "
              f"age≈{face['age']}, "
              f"gender={face['gender']}")
    
    # Save annotated image
    if faces:
        annotated_path = output_dir / f"annotated_{image_path.name}"
        save_annotated_image(image, faces, annotated_path)
        print_info(f"Saved: {annotated_path.name}")
        
        # Save face crops
        if save_crops:
            for face in faces:
                crop_path = output_dir / f"{image_path.stem}_face_{face['face_id']}.jpg"
                save_face_crop(image, face['bbox'], crop_path)
                print_info(f"Saved crop: {crop_path.name}")
    
    return {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'width': width,
        'height': height,
        'faces': faces,
        'processing_time': elapsed
    }


def process_directory(
    pipeline: SimpleCPUFacePipeline,
    directory: Path,
    output_dir: Path,
    save_crops: bool = False
) -> List[Dict]:
    """Process all images in directory."""
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print_error(f"No images found in {directory}")
        return []
    
    print_info(f"Found {len(image_files)} image(s)")
    
    results = []
    total_start = time.time()
    
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        try:
            result = process_single_image(
                pipeline,
                image_path,
                output_dir,
                save_crops
            )
            results.append(result)
            
        except Exception as e:
            print_error(f"Failed to process {image_path.name}: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # Generate and print summary
    summary = generate_summary(results, total_time)
    print_summary(summary)
    
    return results


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test face recognition pipeline on local images (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python test_cpu_local.py photo.jpg
  
  # Process directory
  python test_cpu_local.py /path/to/photos/
  
  # Save face crops
  python test_cpu_local.py /path/to/photos/ --save-crops
  
  # Cluster faces across images
  python test_cpu_local.py /path/to/photos/ --cluster
  
  # Save results to JSON
  python test_cpu_local.py /path/to/photos/ --save-results
  
  # All options combined
  python test_cpu_local.py /path/to/photos/ --save-crops --cluster --save-results
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
        default='test_results',
        help='Output directory (default: test_results)'
    )
    
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save individual face crops'
    )
    
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='Cluster faces to identify same person across images'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON file'
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
        print_error(f"Path not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("Face Recognition Testing (CPU Mode)")
    print_info(f"Input: {input_path}")
    print_info(f"Output: {output_dir}")
    
    # Initialize pipeline
    try:
        pipeline = SimpleCPUFacePipeline()
    except Exception as e:
        print_error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process images
    if input_path.is_file():
        # Single image
        results = [process_single_image(
            pipeline,
            input_path,
            output_dir,
            args.save_crops
        )]
    else:
        # Directory
        results = process_directory(
            pipeline,
            input_path,
            output_dir,
            args.save_crops
        )
    
    if not results:
        print_error("No images processed")
        sys.exit(1)
    
    # Save results to JSON
    if args.save_results:
        json_path = output_dir / 'results.json'
        
        # Remove embeddings from JSON (too large)
        results_for_json = []
        for result in results:
            result_copy = result.copy()
            result_copy['faces'] = [
                {k: v for k, v in face.items() if k != 'embedding'}
                for face in result['faces']
            ]
            results_for_json.append(result_copy)
        
        with open(json_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print_success(f"Results saved to: {json_path}")
    
    # Clustering
    if args.cluster and len(results) > 0:
        print_header("Clustering Faces")
        
        # Collect all faces
        all_faces = []
        for result in results:
            for face in result['faces']:
                face_with_source = face.copy()
                face_with_source['source_image'] = result['image_name']
                all_faces.append(face_with_source)
        
        if all_faces:
            clusters = cluster_faces(all_faces, args.threshold)
            
            if clusters:
                # Save cluster info
                cluster_path = output_dir / 'clusters.json'
                
                clusters_for_json = {}
                for cluster_id, faces in clusters.items():
                    clusters_for_json[str(cluster_id)] = [
                        {
                            'source_image': face['source_image'],
                            'face_id': face['face_id'],
                            'bbox': face['bbox'],
                            'confidence': face['confidence']
                        }
                        for face in faces
                    ]
                
                with open(cluster_path, 'w') as f:
                    json.dump(clusters_for_json, f, indent=2)
                
                print_success(f"Cluster info saved to: {cluster_path}")
    
    # Final summary
    print_header("Complete!")
    print_success(f"Processed {len(results)} image(s)")
    print_success(f"Total faces detected: {sum(len(r['faces']) for r in results)}")
    print_info(f"Results saved in: {output_dir.absolute()}")
    
    print("\n📁 Output files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  • {file.name}")


if __name__ == "__main__":
    main()