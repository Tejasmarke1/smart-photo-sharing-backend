import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Union, List
import numpy as np
import cv2


class TensorRTEmbedder:
    """
    ArcFace embedding model optimized with TensorRT.
    
    Features:
    - INT8/FP16 quantization support
    - Dynamic batching
    - GPU memory optimization
    - Batch inference
    
    Performance targets:
    - Single inference: <5ms on V100
    - Batch-32: <50ms on V100
    """
    
    def __init__(
        self,
        engine_path: str = "models/arcface_fp16.trt",
        embedding_dim: int = 512,
        batch_size: int = 32
    ):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.stream = cuda.Stream()
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Allocate GPU memory for input/output."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for ArcFace.
        
        Args:
            face: RGB aligned face (112, 112, 3)
            
        Returns:
            Preprocessed tensor (1, 3, 112, 112)
        """
        # Convert to float and normalize
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        
        # Transpose to CHW format
        face = np.transpose(face, (2, 0, 1))
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def embed(self, face: np.ndarray) -> np.ndarray:
        """
        Generate embedding for single face.
        
        Args:
            face: Aligned face (112, 112, 3)
            
        Returns:
            L2-normalized embedding (512,)
        """
        # Preprocess
        input_tensor = self.preprocess(face)
        
        # Copy to GPU
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output from GPU
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()
        
        # Extract embedding
        embedding = self.outputs[0]['host'][:self.embedding_dim]
        
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Generate embeddings for batch of faces.
        
        Args:
            faces: List of aligned faces
            
        Returns:
            Embeddings array (N, 512)
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(faces), self.batch_size):
            batch = faces[i:i + self.batch_size]
            
            # Stack and preprocess
            batch_tensor = np.vstack([
                self.preprocess(face) for face in batch
            ])
            
            # Copy to GPU
            np.copyto(self.inputs[0]['host'], batch_tensor.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]['device'],
                self.inputs[0]['host'],
                self.stream
            )
            
            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            # Copy output
            cuda.memcpy_dtoh_async(
                self.outputs[0]['host'],
                self.outputs[0]['device'],
                self.stream
            )
            self.stream.synchronize()
            
            # Extract and normalize embeddings
            batch_embeddings = self.outputs[0]['host'][:len(batch) * self.embedding_dim]
            batch_embeddings = batch_embeddings.reshape(len(batch), self.embedding_dim)
            
            # L2 normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / norms
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine