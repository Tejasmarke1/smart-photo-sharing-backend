# Architecture Overview

## System Components

### API Layer (FastAPI)
- RESTful API endpoints
- JWT-based authentication
- Presigned URL generation for uploads
- WebSocket support for real-time updates

### Worker Layer (Celery)
- Image processing pipeline
- Face detection and embedding
- Clustering and person identification
- Background job processing

### Storage Layer
- PostgreSQL with pgvector for metadata and embeddings
- Redis for caching and job queues
- S3/MinIO for object storage
- Optional: Milvus/FAISS for vector search at scale

### Integration Layer
- Razorpay for payments
- WhatsApp Cloud API for messaging
- CDN for content delivery

## Data Flow

1. **Upload Flow**
   - Client requests presigned URL
   - Direct upload to S3
   - Notification triggers processing pipeline
   - Background workers process images and detect faces

2. **Search Flow**
   - Guest uploads selfie
   - Embedding computed (client or server)
   - Vector similarity search
   - Results returned with thumbnails

3. **Payment Flow**
   - Create Razorpay order
   - Guest completes payment
   - Webhook verification
   - Grant download access
