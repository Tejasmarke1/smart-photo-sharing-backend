from celery import Celery
from celery.schedules import crontab
import os

# Create Celery instance and include explicit task modules
celery_app = Celery(
    'smart_photo_sharing',
    include=[
        'src.tasks.workers.face_processor',
        'src.tasks.workers.search_worker',
        'src.tasks.workers.person_worker',
    ]
)

# Load configuration from environment or default settings
celery_app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://127.0.0.1:6379/1'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://127.0.0.1:6379/2'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes hard limit
)

# Optional: force-import module to ensure task registration in some environments
try:
    from src.tasks.workers import face_processor  # noqa: F401
    from src.tasks.workers import search_worker
    from src.tasks.workers import person_worker  # noqa: F401
except Exception:
    pass


@celery_app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')