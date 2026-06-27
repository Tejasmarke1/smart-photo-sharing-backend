from fastapi import APIRouter
from src.db.base import get_db
from src.api.deps import get_current_user
from src.api.v1.endpoints.faces.helpers import get_pipeline, get_face_or_404, serialize_face

from src.api.v1.endpoints.faces.routes import router as routes_router
from src.api.v1.endpoints.faces.comparison import router as comparison_router
from src.api.v1.endpoints.faces.processing import router as processing_router
from src.api.v1.endpoints.faces.clusters import router as clusters_router

router = APIRouter()

# Include all sub-routers
router.include_router(routes_router)
router.include_router(comparison_router)
router.include_router(processing_router)
router.include_router(clusters_router)
