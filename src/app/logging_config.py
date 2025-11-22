import logging 
import sys
from .config import Settings


def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else "INFO",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    
    