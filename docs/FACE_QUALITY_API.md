# Face Quality Assessment & Advanced Filtering API

## Overview

Production-grade API endpoints for face quality assessment, advanced filtering, duplicate detection, and reprocessing. These endpoints provide comprehensive tools for managing face detection quality and maintaining a clean face database.

## Table of Contents

1. [Quality Assessment Endpoints](#quality-assessment-endpoints)
2. [Face Filtering Endpoints](#face-filtering-endpoints)
3. [Face Reprocessing Endpoints](#face-reprocessing-endpoints)
4. [Authentication & Authorization](#authentication--authorization)
5. [Rate Limiting](#rate-limiting)
6. [Examples](#examples)

---

## Quality Assessment Endpoints

### 1. Get Face Quality Assessment

**Endpoint:** `GET /api/v1/faces/{face_id}/quality`

Get detailed quality metrics for a single face.

**Parameters:**
- `face_id` (path, UUID): Face ID to assess

**Response:** `FaceQualityResponse`
```json
{
  "face_id": "uuid",
  "photo_id": "uuid",
  "blur_score": 0.85,
  "brightness_score": 0.75,
  "confidence": 0.92,
  "overall_quality": 0.81,
  "quality_grade": "B",
  "issues": [],
  "thumbnail_url": "https://..."
}
```

**Quality Grades:**
- **A**: Overall quality ≥ 0.85 (Excellent)
- **B**: Overall quality ≥ 0.70 (Good)
- **C**: Overall quality ≥ 0.55 (Acceptable)
- **D**: Overall quality ≥ 0.40 (Poor)
- **F**: Overall quality < 0.40 (Fail)

**Quality Calculation:**
```
overall_quality = (blur_score × 0.4) + (brightness_score × 0.4) + (confidence × 0.2)
```

**Common Issues:**
- "Image is blurry" - blur_score < 0.5
- "Poor lighting/exposure" - brightness_score < 0.4
- "Overexposed" - brightness_score > 0.9
- "Low detection confidence" - confidence < 0.7

**Rate Limit:** 100 calls per 60 seconds

---

### 2. Batch Quality Check

**Endpoint:** `POST /api/v1/faces/quality-check`

Check quality for multiple faces at once.

**Request Body:** `QualityCheckRequest`
```json
{
  "face_ids": ["uuid1", "uuid2", ...],
  "strict_mode": false
}
```

**Parameters:**
- `face_ids` (array): 1-100 face IDs to check
- `strict_mode` (boolean): Use stricter thresholds (default: false)

**Strict Mode Thresholds:**
- Blur: 0.6 (vs 0.5 normal)
- Brightness: 0.5-0.85 (vs 0.4-0.9 normal)
- Confidence: 0.8 (vs 0.7 normal)

**Response:** Array of `FaceQualityResponse`

**Rate Limit:** 20 calls per 60 seconds

**Use Cases:**
- Identify low-quality faces for deletion
- Batch quality filtering
- Generate quality reports
- Pre-screening before labeling

---

### 3. Find Low-Quality Faces in Album

**Endpoint:** `GET /api/v1/albums/{album_id}/faces/low-quality`

Find all faces in an album that fall below quality thresholds.

**Parameters:**
- `album_id` (path, UUID): Album ID
- `min_overall_quality` (query, float): Minimum quality threshold (0.0-1.0, default: 0.5)
- `limit` (query, int): Max results (1-500, default: 100)

**Response:** Array of `FaceQualityResponse` sorted by quality (worst first)

**Rate Limit:** 50 calls per 60 seconds

**Use Cases:**
- Clean up low-quality detections
- Review problematic faces
- Identify photos needing reprocessing
- Pre-deletion review

---

## Face Filtering Endpoints

### 4. Advanced Face Filtering

**Endpoint:** `POST /api/v1/faces/filter`

Filter faces using complex criteria across multiple dimensions.

**Request Body:** `AdvancedFilterRequest`
```json
{
  "album_ids": ["uuid1", "uuid2"],
  "person_ids": ["uuid3"],
  "min_confidence": 0.7,
  "max_confidence": 1.0,
  "min_blur_score": 0.5,
  "min_brightness_score": 0.4,
  "unlabeled_only": false,
  "has_thumbnail": true,
  "created_after": "2024-01-01T00:00:00Z",
  "created_before": "2024-12-31T23:59:59Z",
  "limit": 100
}
```

**All Parameters (all optional except limit):**
- `album_ids`: Filter by specific albums
- `person_ids`: Filter by labeled persons
- `min_confidence`, `max_confidence`: Confidence range (0.0-1.0)
- `min_blur_score`: Minimum blur score (0.0-1.0)
- `min_brightness_score`: Minimum brightness score (0.0-1.0)
- `unlabeled_only`: Only faces without person labels
- `has_thumbnail`: Filter by thumbnail existence
- `created_after`, `created_before`: Date range (ISO format)
- `limit`: Max results (1-1000, default: 100)

**Response:** Array of `FaceResponse`

**Rate Limit:** 50 calls per 60 seconds

**Permission:** Returns only faces from user's albums (admin sees all)

**Use Cases:**
- Find unlabeled high-quality faces
- Locate faces needing attention
- Generate filtered exports
- Complex search scenarios

---

### 5. Find Duplicate Faces

**Endpoint:** `GET /api/v1/faces/duplicates`

Detect near-duplicate face detections (same person, different photos).

**Parameters:**
- `album_id` (query, UUID, optional): Limit to specific album
- `threshold` (query, float): Similarity threshold (0.8-0.99, default: 0.95)
- `limit` (query, int): Max results (1-200, default: 50)

**Response:** `List[DuplicateFace]`
```json
[
  {
    "face_id_1": "uuid1",
    "face_id_2": "uuid2",
    "similarity_score": 0.97,
    "photo_id_1": "uuid3",
    "photo_id_2": "uuid4",
    "thumbnail_url_1": "https://...",
    "thumbnail_url_2": "https://..."
  }
]
```

**Rate Limit:** 20 calls per 60 seconds

**Caching:** Results cached for 10 minutes

**Use Cases:**
- Identify redundant face detections
- Clean up duplicate entries
- Merge similar faces
- Quality assurance

**Note:** Skips faces from the same photo

---

### 6. Find Face Outliers

**Endpoint:** `GET /api/v1/faces/outliers`

Detect faces that may be mislabeled (low similarity to their person cluster).

**Parameters:**
- `person_id` (query, UUID, optional): Check specific person
- `threshold` (query, float): Similarity threshold (0.3-0.8, default: 0.6)
- `limit` (query, int): Max results (1-200, default: 50)

**Response:** `List[OutlierFace]`
```json
[
  {
    "face_id": "uuid",
    "photo_id": "uuid",
    "person_id": "uuid",
    "person_name": "John Doe",
    "confidence": 0.92,
    "avg_similarity_to_cluster": 0.45,
    "thumbnail_url": "https://..."
  }
]
```

**Response sorted by:** Average similarity (lowest first)

**Rate Limit:** 20 calls per 60 seconds

**Algorithm:**
1. Get all faces for person
2. Calculate average cosine similarity to other faces
3. Flag faces below threshold as outliers

**Use Cases:**
- Find mislabeled faces
- Identify incorrect person assignments
- Quality assurance after clustering
- Manual review candidates

**Minimum Faces:** Requires at least 2 faces per person

---

## Face Reprocessing Endpoints

### 7. Reprocess Single Face

**Endpoint:** `POST /api/v1/faces/{face_id}/reprocess`

Re-run face detection and embedding generation for a specific face.

**Parameters:**
- `face_id` (path, UUID): Face ID to reprocess

**Response:** `ReprocessResponse` (202 Accepted)
```json
{
  "message": "Face reprocessing started",
  "job_id": "celery-task-id",
  "face_ids": ["uuid"],
  "status_url": "/api/v1/jobs/celery-task-id"
}
```

**Rate Limit:** 20 calls per 60 seconds

**Authorization:** Requires `photographer` or `admin` role

**Use Cases:**
- Fix poor quality detections
- Regenerate missing embeddings
- Apply updated models
- Debug detection issues

**Process:**
1. Validates face ownership
2. Enqueues background reprocessing task
3. Returns job ID for status tracking
4. Photo is reprocessed (all faces updated)

---

### 8. Reprocess Album Faces

**Endpoint:** `POST /api/v1/albums/{album_id}/faces/reprocess`

Re-run face detection for all photos in an album.

**Parameters:**
- `album_id` (path, UUID): Album ID

**Request Body:** `BatchReprocessRequest`
```json
{
  "min_quality": 0.5,
  "force_all": false
}
```

**Parameters:**
- `min_quality` (float, optional): Only reprocess faces below this quality (0.0-1.0)
- `force_all` (boolean): Reprocess all faces regardless of quality (default: false)

**Response:** `ReprocessResponse` (202 Accepted)
```json
{
  "message": "Album reprocessing started for 45 photos",
  "job_id": "celery-task-id",
  "status_url": "/api/v1/jobs/celery-task-id"
}
```

**Rate Limit:** 10 calls per 60 seconds

**Authorization:** Requires `photographer` or `admin` role

**Use Cases:**
- Upgrade to new detection models
- Fix batch detection issues
- Improve overall album quality
- Targeted quality improvements

**Behavior:**
- If `force_all=true`: Reprocesses all photos
- If `min_quality` provided: Only reprocesses photos with low-quality faces
- Returns count of photos to reprocess

---

### 9. Batch Delete Faces

**Endpoint:** `POST /api/v1/faces/batch-delete`

Delete multiple faces and optionally their thumbnails.

**Request Body:** `BatchDeleteRequest`
```json
{
  "face_ids": ["uuid1", "uuid2", ...],
  "delete_thumbnails": true
}
```

**Parameters:**
- `face_ids` (array): 1-500 face IDs to delete
- `delete_thumbnails` (boolean): Also delete S3 thumbnails (default: true)

**Response:**
```json
{
  "message": "Faces deleted successfully",
  "deleted_count": 42,
  "deleted_thumbnails": 38,
  "face_ids": ["uuid1", "uuid2", ...]
}
```

**Rate Limit:** 10 calls per 60 seconds

**Authorization:** Requires `photographer` or `admin` role

**Features:**
- Cascades delete to `face_person` mappings
- Optional S3 thumbnail cleanup
- Permission validation per face
- Atomic database transaction

**Use Cases:**
- Remove low-quality detections
- Clean up after reprocessing
- Delete mislabeled faces
- Bulk cleanup operations

**Error Handling:**
- Skips faces user doesn't own
- Logs S3 deletion failures (doesn't fail request)
- Returns actual deleted count

---

## Authentication & Authorization

### Authentication
All endpoints require JWT Bearer token authentication:

```bash
Authorization: Bearer <jwt_token>
```

### Authorization Roles

**Quality Assessment Endpoints:**
- Any authenticated user (can only access own albums)

**Filtering Endpoints:**
- Any authenticated user
- Admin users see all albums
- Regular users see only their albums

**Reprocessing Endpoints:**
- `photographer` role or higher
- `admin` role has full access

**Deletion Endpoints:**
- `photographer` role or higher
- `admin` role has full access

---

## Rate Limiting

Rate limits are applied per endpoint to prevent abuse:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `GET /faces/{id}/quality` | 100 | 60s |
| `POST /faces/quality-check` | 20 | 60s |
| `GET /albums/{id}/faces/low-quality` | 50 | 60s |
| `POST /faces/filter` | 50 | 60s |
| `GET /faces/duplicates` | 20 | 60s |
| `GET /faces/outliers` | 20 | 60s |
| `POST /faces/{id}/reprocess` | 20 | 60s |
| `POST /albums/{id}/faces/reprocess` | 10 | 60s |
| `POST /faces/batch-delete` | 10 | 60s |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704153600
```

---

## Examples

### Example 1: Quality Assessment Workflow

```bash
# Step 1: Get quality for specific face
curl -X GET "https://api.example.com/api/v1/faces/123e4567-e89b-12d3-a456-426614174000/quality" \
  -H "Authorization: Bearer ${TOKEN}"

# Step 2: Find all low-quality faces in album
curl -X GET "https://api.example.com/api/v1/albums/123e4567-e89b-12d3-a456-426614174001/faces/low-quality?min_overall_quality=0.6&limit=50" \
  -H "Authorization: Bearer ${TOKEN}"

# Step 3: Batch check specific faces
curl -X POST "https://api.example.com/api/v1/faces/quality-check" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "face_ids": ["uuid1", "uuid2", "uuid3"],
    "strict_mode": true
  }'
```

### Example 2: Duplicate Detection & Cleanup

```bash
# Step 1: Find duplicates
curl -X GET "https://api.example.com/api/v1/faces/duplicates?album_id=123e4567-e89b-12d3-a456-426614174001&threshold=0.95&limit=100" \
  -H "Authorization: Bearer ${TOKEN}"

# Step 2: Delete duplicate faces
curl -X POST "https://api.example.com/api/v1/faces/batch-delete" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "face_ids": ["uuid1", "uuid2"],
    "delete_thumbnails": true
  }'
```

### Example 3: Outlier Detection & Correction

```bash
# Step 1: Find outliers for a person
curl -X GET "https://api.example.com/api/v1/faces/outliers?person_id=123e4567-e89b-12d3-a456-426614174002&threshold=0.6" \
  -H "Authorization: Bearer ${TOKEN}"

# Step 2: Review outlier and reprocess if needed
curl -X POST "https://api.example.com/api/v1/faces/123e4567-e89b-12d3-a456-426614174003/reprocess" \
  -H "Authorization: Bearer ${TOKEN}"

# Step 3: Check job status
curl -X GET "https://api.example.com/api/v1/jobs/celery-task-id" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Example 4: Advanced Filtering

```bash
# Find high-quality, unlabeled faces created in last 30 days
curl -X POST "https://api.example.com/api/v1/faces/filter" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "min_confidence": 0.8,
    "min_blur_score": 0.7,
    "min_brightness_score": 0.6,
    "unlabeled_only": true,
    "has_thumbnail": true,
    "created_after": "2024-12-01T00:00:00Z",
    "limit": 200
  }'
```

### Example 5: Album Reprocessing

```bash
# Reprocess only low-quality faces
curl -X POST "https://api.example.com/api/v1/albums/123e4567-e89b-12d3-a456-426614174001/faces/reprocess" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "min_quality": 0.6,
    "force_all": false
  }'

# Force reprocess all faces (e.g., after model upgrade)
curl -X POST "https://api.example.com/api/v1/albums/123e4567-e89b-12d3-a456-426614174001/faces/reprocess" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "force_all": true
  }'
```

---

## Error Responses

### Common Error Codes

**400 Bad Request**
```json
{
  "detail": "Invalid date format"
}
```

**403 Forbidden**
```json
{
  "detail": "Access denied"
}
```

**404 Not Found**
```json
{
  "detail": "Face 123e4567-e89b-12d3-a456-426614174000 not found"
}
```

**429 Too Many Requests**
```json
{
  "detail": "Rate limit exceeded"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Face search failed: connection timeout"
}
```

---

## Best Practices

### Quality Assessment
1. **Use strict mode** for critical applications
2. **Set appropriate thresholds** based on your use case
3. **Review borderline cases** (grades C/D) manually
4. **Monitor quality trends** over time

### Duplicate Detection
1. **Start with high threshold** (0.95+) to find exact duplicates
2. **Lower threshold gradually** to find similar faces
3. **Review before deletion** - duplicates may be legitimate
4. **Consider same-album restriction** for large datasets

### Outlier Detection
1. **Use moderate threshold** (0.6-0.7) to start
2. **Review outliers manually** before relabeling
3. **Check confidence scores** - may indicate detection issues
4. **Consider reprocessing** instead of relabeling

### Reprocessing
1. **Test on small subset** before batch operations
2. **Use quality-based filtering** to minimize unnecessary work
3. **Monitor job status** for long-running operations
4. **Schedule during off-peak** hours for large albums

### Batch Operations
1. **Limit batch sizes** to avoid timeouts
2. **Use pagination** for large result sets
3. **Handle partial failures** gracefully
4. **Log operations** for audit trail

---

## Performance Considerations

### Caching
- Duplicate detection: 10-minute cache
- Similar faces: 5-minute cache
- Quality scores: No cache (real-time)

### Background Processing
- Reprocessing uses Celery workers
- Monitor queue depth with `/api/v1/jobs/{job_id}`
- Consider worker scaling for large batches

### Database Queries
- Optimized with proper indexes
- Uses JOIN operations efficiently
- Pagination prevents memory issues

### S3 Operations
- Presigned URLs for thumbnails (1-hour expiry)
- Batch deletions logged but non-blocking
- Retry logic for transient failures

---

## Migration Guide

If upgrading from previous API versions:

1. **Update quality thresholds**: New weighted calculation may affect scores
2. **Review rate limits**: Some limits reduced for expensive operations
3. **Update error handling**: Now uses consistent error format
4. **Check authentication**: Reprocessing requires photographer role
5. **Update frontend**: New response schemas include additional fields

---

## Support

For issues or questions:
- GitHub Issues: [project-repo/issues]
- Email: support@example.com
- Documentation: [docs.example.com]

---

## Changelog

### v1.0.0 (2025-01-01)
- Initial release
- 9 production-grade endpoints
- Quality assessment suite
- Advanced filtering
- Duplicate detection
- Outlier detection
- Reprocessing capabilities
- Batch operations
