# Advanced Clustering & Auto-Labeling API - Quick Reference

## 📋 New Endpoints Summary

### Advanced Clustering (4 endpoints)
| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/albums/{id}/cluster/auto` | POST | Auto-cluster with smart defaults | 5/5min |
| `/albums/{id}/cluster/status` | GET | Get clustering job status | - |
| `/albums/{id}/cluster/results` | GET | Get clustering results | - |
| `/albums/{id}/cluster/review` | POST | Submit cluster review | 20/min |

### Cluster Operations (5 endpoints)
| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/clusters/{id}` | GET | Get cluster details | - |
| `/clusters/{id}/accept` | POST | Accept cluster as person | 50/min |
| `/clusters/{id}/reject` | POST | Reject cluster (noise) | 50/min |
| `/clusters/{id}/split` | POST | Split cluster | 20/min |
| `/clusters/merge` | POST | Merge clusters | 20/min |

### Auto-labeling (4 endpoints)
| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/faces/auto-label` | POST | Auto-label based on persons | 10/min |
| `/faces/label-suggestions` | GET | Get labeling suggestions | - |
| `/faces/{id}/confirm-label` | POST | Confirm auto-label | 100/min |
| `/faces/{id}/reject-label` | POST | Reject auto-label | 100/min |

---

## 🚀 Quick Usage Examples

### 1. Auto-Cluster Workflow
```bash
# Step 1: Start auto-clustering
POST /albums/{id}/cluster/auto
{
  "use_smart_defaults": true
}
# Returns: {"job_id": "...", "status": "pending"}

# Step 2: Check status
GET /albums/{id}/cluster/status?job_id={job_id}

# Step 3: Get results
GET /albums/{id}/cluster/results?job_id={job_id}

# Step 4: Review and accept/reject
POST /albums/{id}/cluster/review
{
  "reviews": [
    {"cluster_id": "...", "action": "accept", "data": {"person_name": "John"}}
  ]
}
```

### 2. Cluster Operations
```bash
# Accept cluster as person
POST /clusters/{id}/accept
{
  "person_name": "John Doe",
  "person_email": "john@example.com"
}

# Split problematic cluster
POST /clusters/{id}/split
{
  "face_groups": [
    ["face1", "face2"],
    ["face3", "face4"]
  ]
}

# Merge similar clusters
POST /clusters/merge
{
  "cluster_ids": ["cluster1", "cluster2"],
  "person_name": "Jane Doe"
}
```

### 3. Auto-Labeling
```bash
# Get suggestions
GET /faces/label-suggestions?album_id={id}&min_confidence=0.8

# Auto-label faces
POST /faces/auto-label
{
  "album_id": "...",
  "min_confidence": 0.8,
  "unlabeled_only": true,
  "max_faces": 100
}

# Confirm suggestion
POST /faces/{face_id}/confirm-label
{
  "person_id": "..."
}
```

---

## 📊 New Schemas

### Auto-Cluster
```python
class AutoClusterRequest:
    use_smart_defaults: bool = True
    min_cluster_size: Optional[int]
    similarity_threshold: Optional[float]
```

### Cluster Detail
```python
class ClusterDetailResponse:
    id: UUID
    album_id: UUID
    cluster_label: int
    size: int
    status: str  # pending, accepted, rejected, split, merged
    avg_similarity: float
    confidence_score: float
    representative_face_id: UUID
    representative_thumbnail_url: str
    face_ids: List[UUID]
    sample_thumbnails: List[str]
    person_id: Optional[UUID]
    person_name: Optional[str]
```

### Label Suggestion
```python
class LabelSuggestion:
    face_id: UUID
    photo_id: UUID
    thumbnail_url: str
    person_id: UUID
    person_name: str
    similarity_score: float
    confidence: str  # high, medium, low
    reasoning: str
```

---

## 🔐 Authentication & Roles

**All clustering/labeling endpoints require:**
- JWT Bearer token
- `photographer` or `admin` role
- Album ownership (or admin)

---

## 🎯 Smart Clustering Features

### Smart Defaults Algorithm
```python
# Based on dataset size:
if faces < 20:
    min_cluster_size = 2
    similarity_threshold = 0.65
elif faces < 100:
    min_cluster_size = 3
    similarity_threshold = 0.70
else:
    min_cluster_size = 5
    similarity_threshold = 0.75
```

### Cluster Status Flow
```
pending → [reviewed] → accepted/rejected/split/merged
```

### Auto-Label Confidence
- **High** (≥0.9): Very high similarity
- **Medium** (0.8-0.9): Good similarity  
- **Low** (0.6-0.8): Moderate similarity

---

## 🗄️ New Database Model

### FaceCluster
```python
class FaceCluster:
    id: UUID
    album_id: UUID
    job_id: str  # Celery task ID
    cluster_label: int  # Algorithm cluster ID
    size: int
    avg_similarity: float
    confidence_score: float
    status: str  # pending, accepted, rejected, split, merged
    representative_face_id: UUID
    person_id: UUID  # If accepted
    merged_into_cluster_id: UUID  # If merged
    reviewed_by_user_id: UUID
    review_notes: str
    face_ids: JSON  # Array of face IDs
```

---

## 📝 Files Modified/Created

### Created
- `src/models/face_cluster.py` - FaceCluster model

### Modified
- `src/schemas/face.py` - Added 12+ new schemas
- `src/api/v1/endpoints/faces.py` - Added 13 new endpoints

### Total
- **Lines Added:** ~1,500 lines
- **Endpoints Before:** 20
- **Endpoints After:** 33
- **New Schemas:** 12
- **New Models:** 1

---

## ⚠️ Important Notes

1. **Clustering is async** - Returns job_id, check status endpoint
2. **Smart defaults recommended** - Algorithm adapts to dataset size
3. **Review required** - Clusters start as "pending" status
4. **Cascading operations** - Accepting cluster creates person + face mappings
5. **Merge/Split creates new clusters** - Original marked as merged/split
6. **Auto-label is ML-based** - Uses cosine similarity on embeddings
7. **Suggestions cached 5 minutes** - Reduces computation overhead

---

## 🔍 Error Handling

### Common Errors
- **404**: Album/Cluster/Person not found
- **403**: Access denied (not album owner)
- **400**: Invalid request (empty groups, wrong actions)
- **429**: Rate limit exceeded

### Validation
- Cluster split: All faces must be accounted for
- Cluster merge: All clusters must be from same album
- Auto-label: Minimum confidence 0.6-0.99

---

## 🚦 Workflow Recommendations

### Best Practice Workflow
1. **Upload photos** to album
2. **Detect faces** (existing endpoint)
3. **Auto-cluster** with smart defaults
4. **Review clusters** - accept/reject/split/merge
5. **Auto-label** remaining faces based on accepted persons
6. **Review suggestions** - confirm/reject
7. **Manual labeling** for edge cases

### For Large Albums (500+ faces)
1. Use smart defaults (auto-adapts parameters)
2. Review in batches (sort by confidence_score)
3. Accept high-confidence clusters first
4. Use auto-label for remaining faces
5. Schedule during off-peak hours

---

## 📈 Performance Considerations

- **Clustering**: O(n²) worst case, cached results
- **Auto-label**: O(nm) for n faces, m persons
- **Suggestions**: Cached 5 minutes
- **Background processing**: Uses Celery workers
- **Database**: Indexed on album_id, job_id, status

---

## 🎉 Summary

**Total New Features:**
- ✅ Smart auto-clustering with adaptive parameters
- ✅ Interactive cluster review workflow
- ✅ Cluster split/merge operations
- ✅ ML-based auto-labeling
- ✅ Confidence-based suggestions
- ✅ Comprehensive status tracking
- ✅ Production-grade error handling
- ✅ Full RBAC and rate limiting

**Ready for Production!** 🚀
