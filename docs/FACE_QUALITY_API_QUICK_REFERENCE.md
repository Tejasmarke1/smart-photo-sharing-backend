# Face Quality API - Quick Reference Card

## 📋 Endpoint Overview

| Category | Endpoint | Method | Purpose | Rate Limit |
|----------|----------|--------|---------|------------|
| **Quality** | `/faces/{id}/quality` | GET | Single face quality | 100/min |
| **Quality** | `/faces/quality-check` | POST | Batch quality check | 20/min |
| **Quality** | `/albums/{id}/faces/low-quality` | GET | Find low-quality faces | 50/min |
| **Filter** | `/faces/filter` | POST | Advanced filtering | 50/min |
| **Filter** | `/faces/duplicates` | GET | Find duplicates | 20/min |
| **Filter** | `/faces/outliers` | GET | Find mislabeled faces | 20/min |
| **Reprocess** | `/faces/{id}/reprocess` | POST | Reprocess one face | 20/min |
| **Reprocess** | `/albums/{id}/faces/reprocess` | POST | Reprocess album | 10/min |
| **Cleanup** | `/faces/batch-delete` | POST | Bulk delete | 10/min |

---

## 🔐 Authentication & Roles

```bash
# All endpoints require JWT Bearer token
Authorization: Bearer <token>

# Role requirements:
- Quality & Filter endpoints: Any authenticated user
- Reprocess endpoints: photographer, admin
- Delete endpoints: photographer, admin
```

---

## 📊 Quality Grades

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A** | ≥ 0.85 | Excellent |
| **B** | ≥ 0.70 | Good |
| **C** | ≥ 0.55 | Acceptable |
| **D** | ≥ 0.40 | Poor |
| **F** | < 0.40 | Fail |

**Formula:**
```
quality = (blur × 0.4) + (brightness × 0.4) + (confidence × 0.2)
```

---

## 🚀 Common Use Cases

### 1. Find & Delete Low-Quality Faces
```bash
# Step 1: Find low-quality faces
GET /albums/{id}/faces/low-quality?min_overall_quality=0.5

# Step 2: Delete them
POST /faces/batch-delete
{
  "face_ids": ["uuid1", "uuid2"],
  "delete_thumbnails": true
}
```

### 2. Detect & Merge Duplicates
```bash
# Step 1: Find duplicates (high similarity)
GET /faces/duplicates?threshold=0.95

# Step 2: Review and delete redundant ones
POST /faces/batch-delete
```

### 3. Fix Mislabeled Faces
```bash
# Step 1: Find outliers
GET /faces/outliers?person_id={id}&threshold=0.6

# Step 2: Reprocess or relabel
POST /faces/{id}/reprocess
```

### 4. Quality-Based Reprocessing
```bash
# Reprocess only low-quality faces
POST /albums/{id}/faces/reprocess
{
  "min_quality": 0.6,
  "force_all": false
}
```

### 5. Advanced Filtering
```bash
# Find high-quality unlabeled faces
POST /faces/filter
{
  "min_confidence": 0.8,
  "min_blur_score": 0.7,
  "unlabeled_only": true,
  "limit": 100
}
```

---

## 📝 Request/Response Examples

### Get Face Quality
```bash
GET /faces/{face_id}/quality

Response:
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

### Batch Quality Check
```bash
POST /faces/quality-check
{
  "face_ids": ["uuid1", "uuid2"],
  "strict_mode": true
}

Response: Array of FaceQualityResponse
```

### Find Duplicates
```bash
GET /faces/duplicates?threshold=0.95&limit=50

Response:
[{
  "face_id_1": "uuid1",
  "face_id_2": "uuid2",
  "similarity_score": 0.97,
  "photo_id_1": "uuid3",
  "photo_id_2": "uuid4",
  "thumbnail_url_1": "https://...",
  "thumbnail_url_2": "https://..."
}]
```

### Find Outliers
```bash
GET /faces/outliers?person_id={id}

Response:
[{
  "face_id": "uuid",
  "photo_id": "uuid",
  "person_id": "uuid",
  "person_name": "John Doe",
  "confidence": 0.92,
  "avg_similarity_to_cluster": 0.45,
  "thumbnail_url": "https://..."
}]
```

### Advanced Filtering
```bash
POST /faces/filter
{
  "album_ids": ["uuid1"],
  "min_confidence": 0.7,
  "unlabeled_only": true,
  "created_after": "2024-01-01T00:00:00Z",
  "limit": 100
}

Response: Array of FaceResponse
```

### Reprocess Face
```bash
POST /faces/{face_id}/reprocess

Response (202):
{
  "message": "Face reprocessing started",
  "job_id": "celery-task-id",
  "face_ids": ["uuid"],
  "status_url": "/api/v1/jobs/{job_id}"
}
```

### Batch Delete
```bash
POST /faces/batch-delete
{
  "face_ids": ["uuid1", "uuid2"],
  "delete_thumbnails": true
}

Response:
{
  "message": "Faces deleted successfully",
  "deleted_count": 2,
  "deleted_thumbnails": 2,
  "face_ids": ["uuid1", "uuid2"]
}
```

---

## ⚠️ Common Issues

### Issue: "Access denied"
**Cause:** User doesn't own the album  
**Fix:** Verify album ownership or use admin role

### Issue: "Rate limit exceeded"
**Cause:** Too many requests in short time  
**Fix:** Wait for rate limit window to reset, implement backoff

### Issue: "Face has no embedding"
**Cause:** Face detection didn't generate embedding  
**Fix:** Use reprocess endpoint to regenerate

### Issue: "No faces found"
**Cause:** Invalid face IDs or no matching results  
**Fix:** Verify face IDs exist and are accessible

### Issue: "Invalid date format"
**Cause:** Date not in ISO 8601 format  
**Fix:** Use format: "2024-01-01T00:00:00Z"

---

## 🎯 Performance Tips

1. **Use batch operations** when possible (quality-check vs individual checks)
2. **Enable caching** for duplicate/similarity searches (cached 10 minutes)
3. **Limit result sets** to avoid timeouts (use pagination)
4. **Use background jobs** for expensive operations (reprocessing)
5. **Filter before processing** to reduce workload
6. **Monitor rate limits** to avoid throttling
7. **Schedule large operations** during off-peak hours

---

## 🔍 Query Parameters Guide

### Quality Filtering
- `min_blur_score`: 0.0-1.0 (default: none)
- `min_brightness_score`: 0.0-1.0 (default: none)
- `min_confidence`: 0.0-1.0 (default: none)
- `min_overall_quality`: 0.0-1.0 (default: 0.5)

### Similarity Thresholds
- **Exact duplicates**: 0.95-0.99
- **Very similar**: 0.85-0.95
- **Similar**: 0.70-0.85
- **Somewhat similar**: 0.60-0.70
- **Different**: < 0.60

### Limits
- Quality check: 1-100 faces
- Filter results: 1-1000 faces
- Duplicates: 1-200 pairs
- Outliers: 1-200 faces
- Batch delete: 1-500 faces

---

## 🧪 Testing Commands

```bash
# Set environment
export TOKEN="your-jwt-token"
export BASE_URL="http://localhost:8000/api/v1"

# Test quality assessment
curl -X GET "$BASE_URL/faces/{face_id}/quality" \
  -H "Authorization: Bearer $TOKEN"

# Test batch quality check
curl -X POST "$BASE_URL/faces/quality-check" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"face_ids":["uuid1"],"strict_mode":false}'

# Test duplicate detection
curl -X GET "$BASE_URL/faces/duplicates?threshold=0.95" \
  -H "Authorization: Bearer $TOKEN"

# Test outlier detection
curl -X GET "$BASE_URL/faces/outliers?threshold=0.6" \
  -H "Authorization: Bearer $TOKEN"

# Test advanced filtering
curl -X POST "$BASE_URL/faces/filter" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"min_confidence":0.8,"unlabeled_only":true}'

# Test reprocessing
curl -X POST "$BASE_URL/faces/{face_id}/reprocess" \
  -H "Authorization: Bearer $TOKEN"

# Test batch delete
curl -X POST "$BASE_URL/faces/batch-delete" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"face_ids":["uuid1"],"delete_thumbnails":true}'
```

---

## 📚 Related Documentation

- **Full API Documentation:** [docs/FACE_QUALITY_API.md](FACE_QUALITY_API.md)
- **Implementation Summary:** [docs/IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **OpenAPI/Swagger:** `/docs`
- **ReDoc:** `/redoc`

---

## 💡 Best Practices

1. ✅ **Always check permissions** before batch operations
2. ✅ **Use strict mode** for critical applications
3. ✅ **Review outliers manually** before relabeling
4. ✅ **Test on small datasets** before large batches
5. ✅ **Monitor job status** for async operations
6. ✅ **Handle errors gracefully** with retries
7. ✅ **Log all operations** for audit trail
8. ✅ **Cache results** when appropriate
9. ✅ **Use pagination** for large result sets
10. ✅ **Schedule heavy operations** off-peak

---

## 🆘 Support

- **Issues:** GitHub Issues
- **Documentation:** `/docs` endpoints
- **Logs:** Check server logs for details
- **Monitoring:** Track job status at `/api/v1/jobs/{job_id}`

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-01  
**Status:** ✅ Production Ready
