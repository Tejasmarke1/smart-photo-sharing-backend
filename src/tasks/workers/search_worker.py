"""
FAISS Index Management Workers
==============================

Celery tasks to:
- Build FAISS index
- Rebuild index from DB (global or per-album)
- Persist index to disk / S3
- Warm cache by loading index into memory
- Album-level reindexing

Uses a global GLOBAL_FAISS_ENGINE instance (not the full pipeline).
"""

from typing import Optional, Dict, Any
import os
import logging

import numpy as np
from celery import Task
from celery.signals import worker_ready

from src.tasks.celery_app import celery_app
from src.db.base import SessionLocal
from src.services.face.pipeline import FlexibleFAISSVectorSearch
from src.services.storage.s3 import S3Service
from src.models.face import Face	
from src.models.photo import Photo

logger = logging.getLogger(__name__)


# =============================================================================
# Global FAISS Engine (Shared across all tasks)
# =============================================================================

GLOBAL_FAISS_ENGINE: Optional[FlexibleFAISSVectorSearch] = None


def init_global_faiss():
	"""Initialize the global FAISS engine."""
	global GLOBAL_FAISS_ENGINE
	if GLOBAL_FAISS_ENGINE is None:
		logger.info("🔧 Initializing global FAISS engine...")
		GLOBAL_FAISS_ENGINE = FlexibleFAISSVectorSearch(device='cpu')
	return GLOBAL_FAISS_ENGINE


@worker_ready.connect
def warm_index_at_startup(**kwargs):
	"""Warm FAISS cache once when worker starts."""
	try:
		index_path = os.getenv("FAISS_INDEX_PATH", os.path.join("data", "models", "faiss", "index"))
		if os.path.exists(f"{index_path}.faiss"):
			logger.info(f"🔥 Warming FAISS cache from {index_path}...")
			engine = init_global_faiss()
			engine.load(index_path)
			ntotal = getattr(engine.index, "ntotal", 0)
			logger.info(f"✅ FAISS cache warmed with {ntotal} faces")
		else:
			logger.warning(f"⚠️ No FAISS index found at {index_path} - skipping warmup")
	except Exception as e:
		logger.error(f"❌ Failed to warm FAISS cache on startup: {e}", exc_info=True)


# =============================================================================
# Base Task Class
# =============================================================================

class SearchTask(Task):
	"""Base task class for search/index operations using global FAISS."""

	_s3_service: Optional[S3Service] = None

	@property
	def faiss(self) -> FlexibleFAISSVectorSearch:
		"""Get the global FAISS engine."""
		return init_global_faiss()

	@property
	def s3_service(self) -> S3Service:
		"""Lazy-load S3 service."""
		if self._s3_service is None:
			self._s3_service = S3Service()
		return self._s3_service

	def get_db(self):
		"""Get a fresh DB session."""
		return SessionLocal()

# =============================================================================
# Helpers
# =============================================================================

def _default_index_path(album_id: Optional[str] = None) -> str:
	"""Get default local path for FAISS index."""
	base = os.getenv("FAISS_INDEX_PATH", os.path.join("data", "models", "faiss", "index"))
	if album_id:
		return f"{base}_{album_id}"
	return base


def _default_s3_prefix(album_id: Optional[str] = None) -> str:
	"""Get default S3 prefix for FAISS index."""
	prefix = os.getenv("FAISS_INDEX_S3_PREFIX", "faiss/index")
	if album_id:
		return f"{prefix}/{album_id}"
	return prefix


# =============================================================================
# Tasks
# =============================================================================

@celery_app.task(bind=True, base=SearchTask, name="tasks.rebuild_index_from_db", track_started=True)
def rebuild_index_from_db_task(self, album_id: Optional[str] = None) -> Dict[str, Any]:
	"""Rebuild FAISS index from DB embeddings, optionally scoped to an album."""
	import uuid as uuid_lib
	db = self.get_db()
	try:
		scope = f"album {album_id}" if album_id else "all faces"
		self.update_state(state="PROCESSING", meta={"status": f"Rebuilding index for {scope}", "progress": 20})
		
		# Query faces with embeddings directly from DB
		if album_id:
			faces_q = (
				db.query(Face)
				.join(Photo)
				.filter(Photo.album_id == uuid_lib.UUID(album_id))
				.filter(Face.embedding.isnot(None))
			)
		else:
			faces_q = db.query(Face).filter(Face.embedding.isnot(None))

		faces = faces_q.all()
		
		if not faces:
			logger.info(f"No faces with embeddings found for {scope}")
			# Reset engine to empty index
			self.faiss.index = FlexibleFAISSVectorSearch(device='cpu').index
			self.faiss.id_map = {}
			return {"status": "completed", "faces_indexed": 0, "album_id": album_id}

		# Build embeddings matrix and IDs
		embs = []
		face_ids = []
		for f in faces:
			try:
				vec = np.array(f.embedding, dtype=np.float32)
				if vec.ndim == 1:
					embs.append(vec)
					face_ids.append(str(f.id))
			except Exception as e:
				logger.warning(f"Failed to process embedding for face {f.id}: {e}")
				continue

		if not embs:
			logger.warning(f"No valid embeddings found for {scope}")
			return {"status": "completed", "faces_indexed": 0, "album_id": album_id}

		# Stack embeddings and add to FAISS
		emb_matrix = np.vstack(embs).astype(np.float32)
		logger.info(f"Training FAISS index with {len(embs)} embeddings...")
		self.faiss.index.train(emb_matrix)
		self.faiss.index.add(emb_matrix)
		
		# Store ID mappings
		self.faiss.id_map = {i: face_id for i, face_id in enumerate(face_ids)}
		
		self.update_state(state="PROCESSING", meta={"status": f"Rebuilt {len(face_ids)} faces", "progress": 100})
		
		return {
			"status": "completed",
			"faces_indexed": len(face_ids),
			"album_id": album_id
		}
		
	except Exception as e:
		logger.error(f"Rebuild index failed for {album_id}: {e}", exc_info=True)
		raise
	finally:
		db.close()


@celery_app.task(bind=True, base=SearchTask, name="tasks.persist_index", track_started=True)
def persist_index_task(
	self,
	album_id: Optional[str] = None,
	to_s3: bool = True,
	local_path: Optional[str] = None,
	s3_prefix: Optional[str] = None
) -> Dict[str, Any]:
	"""Persist the global FAISS index to disk and optionally upload to S3."""
	
	path = local_path or _default_index_path(album_id)
	s3_key_prefix = s3_prefix or _default_s3_prefix(album_id)

	try:
		self.update_state(state="PROCESSING", meta={"status": "Saving FAISS index to disk", "progress": 20})
		
		# Ensure directory exists
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		# Save global FAISS engine using instance method
		self.faiss.save(path)
		logger.info(f"✅ Saved FAISS index to {path}")
		
		uploaded = False
		if to_s3:
			self.update_state(state="PROCESSING", meta={"status": "Uploading to S3", "progress": 60})
			
			# Upload .faiss and .pkl files
			faiss_path = f"{path}.faiss"
			pkl_path = f"{path}.pkl"
			
			if os.path.exists(faiss_path):
				with open(faiss_path, "rb") as f:
					self.s3_service.upload_file(
						f.read(),
						f"{s3_key_prefix}.faiss",
						content_type="application/octet-stream"
					)
				logger.info(f"✅ Uploaded {faiss_path} to S3")
			
			if os.path.exists(pkl_path):
				with open(pkl_path, "rb") as f:
					self.s3_service.upload_file(
						f.read(),
						f"{s3_key_prefix}.pkl",
						content_type="application/octet-stream"
					)
				logger.info(f"✅ Uploaded {pkl_path} to S3")
			
			uploaded = True

		self.update_state(state="PROCESSING", meta={"status": "Persist complete", "progress": 100})
		
		return {
			"status": "completed",
			"local_path": path,
			"s3_prefix": s3_key_prefix if to_s3 else None,
			"uploaded_to_s3": uploaded,
		}
		
	except Exception as e:
		logger.error(f"Persist index failed: {e}", exc_info=True)
		raise


@celery_app.task(bind=True, base=SearchTask, name="tasks.warm_faiss_cache", track_started=True)
def warm_faiss_cache_task(
	self,
	album_id: Optional[str] = None,
	from_s3: bool = False,
	local_path: Optional[str] = None,
	s3_prefix: Optional[str] = None
) -> Dict[str, Any]:
	"""Warm cache by loading FAISS index into memory from disk or S3."""
	
	path = local_path or _default_index_path(album_id)
	s3_key_prefix = s3_prefix or _default_s3_prefix(album_id)

	try:
		self.update_state(state="PROCESSING", meta={"status": "Warming FAISS cache", "progress": 10})

		if from_s3:
			self.update_state(state="PROCESSING", meta={"status": "Downloading from S3", "progress": 30})
			
			# Download index files from S3
			os.makedirs(os.path.dirname(path), exist_ok=True)
			
			faiss_bytes = self.s3_service.download_file(f"{s3_key_prefix}.faiss")
			pkl_bytes = self.s3_service.download_file(f"{s3_key_prefix}.pkl")
			
			if not faiss_bytes or not pkl_bytes:
				raise ValueError(f"Missing FAISS index files in S3 at {s3_key_prefix}")
			
			with open(f"{path}.faiss", "wb") as f:
				f.write(faiss_bytes)
			with open(f"{path}.pkl", "wb") as f:
				f.write(pkl_bytes)
			logger.info(f"✅ Downloaded FAISS index from S3 to {path}")

		self.update_state(state="PROCESSING", meta={"status": "Loading into memory", "progress": 60})
		
		# Load into global FAISS engine
		GLOBAL_FAISS_ENGINE.load(path)
		ntotal = getattr(GLOBAL_FAISS_ENGINE.index, "ntotal", 0)
		logger.info(f"✅ Loaded {ntotal} faces into FAISS cache")

		self.update_state(state="PROCESSING", meta={"status": "Cache warmed", "faces_indexed": ntotal, "progress": 100})
		
		return {
			"status": "completed",
			"faces_indexed": int(ntotal),
			"album_id": album_id
		}
		
	except Exception as e:
		logger.error(f"Warm cache failed: {e}", exc_info=True)
		raise


@celery_app.task(bind=True, base=SearchTask, name="tasks.reindex_album", track_started=True)
def reindex_album_task(
	self,
	album_id: str,
	persist: bool = True,
	to_s3: bool = True
) -> Dict[str, Any]:
	"""Album-level reindexing: rebuild index for a single album and persist it."""
	import uuid as uuid_lib
	db = self.get_db()
	try:
		self.update_state(state="PROCESSING", meta={"status": f"Reindexing album {album_id}", "progress": 15})
		
		# Query faces with embeddings directly from DB
		faces_q = (
			db.query(Face)
			.join(Photo)
			.filter(Photo.album_id == uuid_lib.UUID(album_id))
			.filter(Face.embedding.isnot(None))
		)

		faces = faces_q.all()
		
		if not faces:
			logger.info(f"No faces with embeddings found for album {album_id}")
			self.faiss.index = FlexibleFAISSVectorSearch(device='cpu').index
			self.faiss.id_map = {}
			return {"status": "completed", "album_id": album_id, "faces_indexed": 0}

		# Build embeddings matrix and IDs
		embs = []
		face_ids = []
		for f in faces:
			try:
				vec = np.array(f.embedding, dtype=np.float32)
				if vec.ndim == 1:
					embs.append(vec)
					face_ids.append(str(f.id))
			except Exception as e:
				logger.warning(f"Failed to process embedding for face {f.id}: {e}")
				continue

		if not embs:
			logger.warning(f"No valid embeddings found for album {album_id}")
			return {"status": "completed", "album_id": album_id, "faces_indexed": 0}

		# Stack embeddings and add to FAISS
		emb_matrix = np.vstack(embs).astype(np.float32)
		logger.info(f"Training FAISS index for album {album_id} with {len(embs)} embeddings...")
		self.faiss.index.train(emb_matrix)
		self.faiss.index.add(emb_matrix)
		
		# Store ID mappings
		self.faiss.id_map = {i: face_id for i, face_id in enumerate(face_ids)}
		
		self.update_state(state="PROCESSING", meta={"status": f"Indexed {len(face_ids)} faces", "progress": 60})
		
		result = {"status": "completed", "album_id": album_id, "faces_indexed": len(face_ids)}
		
		if persist:
			self.update_state(state="PROCESSING", meta={"status": "Persisting", "progress": 80})
			# Queue persist task asynchronously
			persist_index_task.apply_async(
				kwargs={"album_id": album_id, "to_s3": to_s3}
			)
			result["persist_queued"] = True
		
		self.update_state(state="PROCESSING", meta={"status": "Reindex complete", "progress": 100})
		
		return result
		
	except Exception as e:
		logger.error(f"Album reindex failed for {album_id}: {e}", exc_info=True)
		raise
	finally:
		db.close()


@celery_app.task(bind=True, base=SearchTask, name="tasks.build_faiss_index", track_started=True)
def build_faiss_index_task(self, persist: bool = True, to_s3: bool = True) -> Dict[str, Any]:
	"""Build a global FAISS index from all faces in DB and optionally persist it."""
	db = self.get_db()
	try:
		self.update_state(state="PROCESSING", meta={"status": "Building global index", "progress": 15})
		
		# Query faces with embeddings directly from DB
		faces_q = db.query(Face).filter(Face.embedding.isnot(None))
		faces = faces_q.all()
		
		if not faces:
			logger.info("No faces with embeddings found in DB")
			self.faiss.index = FlexibleFAISSVectorSearch(device='cpu').index
			self.faiss.id_map = {}
			return {"status": "completed", "faces_indexed": 0, "album_id": None}

		# Build embeddings matrix and IDs
		embs = []
		face_ids = []
		for f in faces:
			try:
				vec = np.array(f.embedding, dtype=np.float32)
				if vec.ndim == 1:
					embs.append(vec)
					face_ids.append(str(f.id))
			except Exception as e:
				logger.warning(f"Failed to process embedding for face {f.id}: {e}")
				continue

		if not embs:
			logger.warning("No valid embeddings found in DB")
			return {"status": "completed", "faces_indexed": 0, "album_id": None}

		# Stack embeddings and add to FAISS
		emb_matrix = np.vstack(embs).astype(np.float32)
		logger.info(f"Training FAISS index with {len(embs)} embeddings...")
		self.faiss.index.train(emb_matrix)
		self.faiss.index.add(emb_matrix)
		
		# Store ID mappings
		self.faiss.id_map = {i: face_id for i, face_id in enumerate(face_ids)}
		
		self.update_state(state="PROCESSING", meta={"status": f"Indexed {len(face_ids)} faces", "progress": 50})
		
		result = {"status": "completed", "faces_indexed": len(face_ids), "album_id": None}
		
		if persist:
			self.update_state(state="PROCESSING", meta={"status": "Persisting", "progress": 75})
			# Queue persist task asynchronously
			persist_index_task.apply_async(
				kwargs={"album_id": None, "to_s3": to_s3}
			)
			result["persist_queued"] = True
		
		self.update_state(state="PROCESSING", meta={"status": "Build complete", "progress": 100})
		
		return result
		
	except Exception as e:
		logger.error(f"Build global index failed: {e}", exc_info=True)
		raise
	finally:
		db.close()

