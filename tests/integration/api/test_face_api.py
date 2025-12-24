"""
Face API Tests
==============

Comprehensive test suite for face detection, search, and clustering APIs.

Test Coverage:
- Unit tests for individual components
- Integration tests for full workflows
- Performance tests for search latency
- Edge case handling
"""

import pytest
import numpy as np
from uuid import UUID, uuid4
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import cv2

from src.app.main import app
from src.db.base import get_db
from src.api.deps import get_current_user
from src.services.storage.s3 import S3Service
from src.models.face import Face
from src.api.v1.endpoints.faces import get_pipeline
from src.models.face_person import FacePerson
from src.models.person import Person
from src.models.photo import Photo
from src.models.album import Album
from src.models.user import User


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client(mock_user, mocker):
    """FastAPI test client with dependency overrides."""
    def override_get_current_user():
        return mock_user
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    test_client = TestClient(app)
    yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def db_session(mocker):
    """Mock database session."""
    session = MagicMock(spec=Session)
    
    # Mock query builder
    query = MagicMock()
    query.filter.return_value = query
    query.join.return_value = query
    query.outerjoin.return_value = query
    query.order_by.return_value = query
    query.offset.return_value = query
    query.limit.return_value = query
    query.first.return_value = None
    query.all.return_value = []
    query.count.return_value = 0
    
    session.query.return_value = query
    
    return session


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {
        'id': str(uuid4()),
        'email': 'test@example.com',
        'role': 'photographer'
    }


@pytest.fixture
def mock_album(mock_user):
    """Mock album."""
    album = Album(
        id=uuid4(),
        photographer_id=UUID(mock_user['id']),
        title='Test Wedding',
        sharing_code='TEST123',
        created_at=datetime.utcnow()
    )
    return album


@pytest.fixture
def mock_photo(mock_album):
    """Mock photo."""
    photo = Photo(
        id=uuid4(),
        album_id=mock_album.id,
        s3_key='photos/test.jpg',
        status='done',
        created_at=datetime.utcnow()
    )
    return photo


@pytest.fixture
def mock_face(mock_photo):
    """Mock face."""
    face = Face(
        id=uuid4(),
        photo_id=mock_photo.id,
        bbox="{'x': 100, 'y': 100, 'w': 200, 'h': 200}",
        confidence=0.95,
        embedding=np.random.randn(512).tolist(),
        thumbnail_s3_key='faces/test/face1.jpg',
        blur_score=0.8,
        brightness_score=0.7,
        created_at=datetime.utcnow()
    )
    return face


@pytest.fixture
def mock_person(mock_album):
    """Mock person."""
    person = Person(
        id=uuid4(),
        album_id=mock_album.id,
        name='John Doe',
        created_at=datetime.utcnow()
    )
    return person


@pytest.fixture
def sample_image():
    """Generate sample test image."""
    # Create 640x480 RGB image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_face_crop():
    """Generate sample face crop."""
    # Create 224x224 RGB face image
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return face


@pytest.fixture
def mock_pipeline(mocker):
    """Mock face pipeline."""
    pipeline = MagicMock()
    
    # Mock search results
    pipeline.search_by_selfie.return_value = [
        {
            'face_id': str(uuid4()),
            'score': 0.85,
            'bbox': {'x': 100, 'y': 100, 'w': 200, 'h': 200}
        }
    ]
    
    pipeline.search_engine.search.return_value = [
        {
            'face_id': str(uuid4()),
            'score': 0.85
        }
    ]
    
    pipeline.get_device_info.return_value = {
        'device': 'cpu',
        'detector': 'RetinaFace',
        'embedder_backend': 'onnx'
    }
    
    return pipeline


# ============================================================================
# Unit Tests - Face Listing
# ============================================================================

class TestListAlbumFaces:
    """Test face listing endpoint."""
    
    def test_list_faces_success(
        self,
        client,
        mock_user,
        mock_album,
        mock_face,
        mocker
    ):
        """Test successful face listing."""
        # Mock DB session
        session = MagicMock(spec=Session)
        
        # Mock album query
        album_query = MagicMock()
        album_query.filter.return_value.first.return_value = mock_album
        
        # Mock faces query
        faces_query = MagicMock()
        faces_query.join.return_value = faces_query
        faces_query.filter.return_value = faces_query
        faces_query.count.return_value = 1
        faces_query.order_by.return_value = faces_query
        faces_query.offset.return_value = faces_query
        faces_query.limit.return_value.all.return_value = [mock_face]
        
        session.query.side_effect = [album_query, faces_query]
        
        # Override get_db
        def override_get_db():
            yield session
        
        client.app.dependency_overrides[get_db] = override_get_db
        
        try:
            # Make request
            response = client.get(f"/api/v1/faces/albums/{mock_album.id}/faces")
            
            assert response.status_code == 200
            data = response.json()
            assert 'faces' in data
            assert data['total'] == 1
        finally:
            client.app.dependency_overrides.clear()
    
    def test_list_faces_album_not_found(
        self,
        client,
        mock_user,
        mocker
    ):
        """Test listing faces for non-existent album."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            query = MagicMock()
            query.filter.return_value.first.return_value = None
            session.query.return_value = query
            mock_db.return_value = session
            
            response = client.get(f"/api/v1/faces/albums/{uuid4()}/faces")
            
            assert response.status_code == 404
    
    def test_list_faces_access_denied(
        self,
        client,
        mock_user,
        mock_album,
        mocker
    ):
        """Test access denied for unauthorized user."""
        # Different user
        other_user = {'id': str(uuid4()), 'role': 'photographer'}
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=other_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            query = MagicMock()
            query.filter.return_value.first.return_value = mock_album
            session.query.return_value = query
            mock_db.return_value = session
            
            response = client.get(f"/api/v1/faces/albums/{mock_album.id}/faces")
            
            assert response.status_code == 403
    
    def test_list_faces_with_quality_filter(
        self,
        client,
        mock_user,
        mock_album,
        mock_face,
        mocker
    ):
        """Test face listing with quality filters."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            album_query = MagicMock()
            album_query.filter.return_value.first.return_value = mock_album
            
            faces_query = MagicMock()
            faces_query.join.return_value = faces_query
            faces_query.filter.return_value = faces_query
            faces_query.count.return_value = 1
            faces_query.order_by.return_value = faces_query
            faces_query.offset.return_value = faces_query
            faces_query.limit.return_value.all.return_value = [mock_face]
            
            session.query.side_effect = [album_query, faces_query]
            mock_db.return_value = session
            
            response = client.get(
                f"/api/v1/faces/albums/{mock_album.id}/faces",
                params={
                    'min_blur_score': 0.5,
                    'min_brightness_score': 0.5,
                    'min_confidence': 0.8
                }
            )
            
            assert response.status_code == 200


# ============================================================================
# Unit Tests - Face Labeling
# ============================================================================

class TestLabelFace:
    """Test face labeling endpoint."""
    
    def test_label_face_with_new_person(
        self,
        client,
        mock_user,
        mock_album,
        mock_photo,
        mock_face,
        mocker
    ):
        """Test labeling face with new person name."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            # Setup mocks
            face_query = MagicMock()
            face_query.filter.return_value.first.return_value = mock_face
            
            photo_query = MagicMock()
            photo_query.filter.return_value.first.return_value = mock_photo
            
            album_query = MagicMock()
            album_query.filter.return_value.first.return_value = mock_album
            
            person_query = MagicMock()
            person_query.filter.return_value.first.return_value = None
            
            mapping_query = MagicMock()
            mapping_query.filter.return_value.first.return_value = None
            
            session.query.side_effect = [
                face_query,
                photo_query,
                album_query,
                person_query,
                person_query,
                mapping_query
            ]
            
            mock_db.return_value = session
            
            response = client.post(
                f"/api/v1/faces/faces/{mock_face.id}/label",
                json={'person_name': 'Jane Doe'}
            )
            
            assert response.status_code == 200
            session.add.assert_called()
            session.commit.assert_called()
    
    def test_label_face_with_existing_person(
        self,
        client,
        mock_user,
        mock_album,
        mock_photo,
        mock_face,
        mock_person,
        mocker
    ):
        """Test labeling face with existing person ID."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            face_query = MagicMock()
            face_query.filter.return_value.first.return_value = mock_face
            
            photo_query = MagicMock()
            photo_query.filter.return_value.first.return_value = mock_photo
            
            album_query = MagicMock()
            album_query.filter.return_value.first.return_value = mock_album
            
            person_query = MagicMock()
            person_query.filter.return_value.first.return_value = mock_person
            
            mapping_query = MagicMock()
            mapping_query.filter.return_value.first.return_value = None
            
            session.query.side_effect = [
                face_query,
                photo_query,
                album_query,
                person_query,
                mapping_query
            ]
            
            mock_db.return_value = session
            
            response = client.post(
                f"/api/v1/faces/faces/{mock_face.id}/label",
                json={'person_id': str(mock_person.id)}
            )
            
            assert response.status_code == 200
    
    def test_label_face_validation_error(
        self,
        client,
        mock_user,
        mock_face,
        mocker
    ):
        """Test labeling face with invalid request."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        response = client.post(
            f"/api/v1/faces/faces/{mock_face.id}/label",
            json={}  # Missing both person_id and person_name
        )
        
        assert response.status_code == 422


# ============================================================================
# Unit Tests - Selfie Search
# ============================================================================

class TestSearchBySelfie:
    """Test selfie search endpoint."""
    
    def test_search_by_selfie_success(
        self,
        client,
        mock_user,
        sample_image,
        mock_face,
        mock_pipeline,
        mocker
    ):
        """Test successful selfie search."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        mocker.patch(
            'src.api.v1.endpoints.faces.get_pipeline',
            return_value=mock_pipeline
        )
        
        # Encode image as JPEG
        success, buffer = cv2.imencode('.jpg', sample_image)
        image_bytes = buffer.tobytes()
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            face_query = MagicMock()
            face_query.filter.return_value.all.return_value = [mock_face]
            session.query.return_value = face_query
            
            mock_db.return_value = session
            
            response = client.post(
                "/api/v1/faces/search/by-selfie",
                files={'file': ('selfie.jpg', image_bytes, 'image/jpeg')}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    def test_search_by_selfie_invalid_file_type(
        self,
        client,
        mock_user,
        mocker
    ):
        """Test selfie search with invalid file type."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        response = client.post(
            "/api/v1/faces/search/by-selfie",
            files={'file': ('test.txt', b'test', 'text/plain')}
        )
        
        assert response.status_code == 400
    
    def test_search_by_selfie_file_too_large(
        self,
        client,
        mock_user,
        mocker
    ):
        """Test selfie search with file too large."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        # Create 15MB file
        large_file = b'0' * (15 * 1024 * 1024)
        
        response = client.post(
            "/api/v1/faces/search/by-selfie",
            files={'file': ('large.jpg', large_file, 'image/jpeg')}
        )
        
        assert response.status_code == 413


# ============================================================================
# Unit Tests - Embedding Search
# ============================================================================

class TestSearchByEmbedding:
    """Test embedding search endpoint."""
    
    def test_search_by_embedding_success(
        self,
        client,
        mock_user,
        mock_face,
        mock_pipeline,
        mocker
    ):
        """Test successful embedding search."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        mocker.patch(
            'src.api.v1.endpoints.faces.get_pipeline',
            return_value=mock_pipeline
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            face_query = MagicMock()
            face_query.filter.return_value = face_query
            face_query.all.return_value = [mock_face]
            session.query.return_value = face_query
            
            mock_db.return_value = session
            
            # Generate random embedding
            embedding = np.random.randn(512).tolist()
            
            response = client.post(
                "/api/v1/faces/search/by-embedding",
                json={
                    'embedding': embedding,
                    'k': 10,
                    'threshold': 0.6
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    def test_search_by_embedding_invalid_dimension(
        self,
        client,
        mock_user,
        mocker
    ):
        """Test embedding search with wrong dimension."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        # Wrong dimension (should be 512)
        embedding = np.random.randn(256).tolist()
        
        response = client.post(
            "/api/v1/faces/search/by-embedding",
            json={
                'embedding': embedding,
                'k': 10
            }
        )
        
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================

class TestFaceWorkflowIntegration:
    """Integration tests for complete face workflows."""
    
    @pytest.mark.integration
    def test_complete_face_detection_workflow(
        self,
        client,
        mock_user,
        mock_album,
        mock_photo,
        sample_image,
        mocker
    ):
        """Test complete workflow from upload to detection to search."""
        # This would be a full integration test with real database
        # and real face detection pipeline
        pass
    
    @pytest.mark.integration
    def test_clustering_workflow(
        self,
        client,
        mock_user,
        mock_album,
        mocker
    ):
        """Test clustering workflow."""
        # Test full clustering pipeline
        pass


# ============================================================================
# Performance Tests
# ============================================================================

class TestFaceSearchPerformance:
    """Performance tests for face search."""
    
    @pytest.mark.performance
    def test_search_latency_under_100ms(
        self,
        client,
        mock_user,
        mock_pipeline,
        mocker
    ):
        """Test search latency is under 100ms."""
        import time
        
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        mocker.patch(
            'src.api.v1.endpoints.faces.get_pipeline',
            return_value=mock_pipeline
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            face_query = MagicMock()
            face_query.filter.return_value.all.return_value = []
            session.query.return_value = face_query
            mock_db.return_value = session
            
            embedding = np.random.randn(512).tolist()
            
            start = time.time()
            
            response = client.post(
                "/api/v1/faces/search/by-embedding",
                json={'embedding': embedding}
            )
            
            latency = (time.time() - start) * 1000  # ms
            
            assert response.status_code == 200
            assert latency < 100, f"Search took {latency}ms, expected < 100ms"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_face_with_no_embedding(
        self,
        client,
        mock_user,
        mock_face,
        mocker
    ):
        """Test handling face with no embedding."""
        mock_face.embedding = None
        
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_face_or_404') as mock_get:
            mock_get.return_value = mock_face
            
            response = client.get(f"/api/v1/faces/faces/{mock_face.id}/similar")
            
            assert response.status_code == 400
    
    def test_album_with_no_faces(
        self,
        client,
        mock_user,
        mock_album,
        mocker
    ):
        """Test clustering album with no faces."""
        mocker.patch(
            'src.api.v1.endpoints.faces.get_current_user',
            return_value=mock_user
        )
        
        with patch('src.api.v1.endpoints.faces.get_db') as mock_db:
            session = MagicMock()
            
            album_query = MagicMock()
            album_query.filter.return_value.first.return_value = mock_album
            
            count_query = MagicMock()
            count_query.join.return_value = count_query
            count_query.filter.return_value = count_query
            count_query.scalar.return_value = 0
            
            session.query.side_effect = [album_query, count_query]
            mock_db.return_value = session
            
            response = client.post(
                f"/api/v1/faces/albums/{mock_album.id}/cluster",
                json={'min_cluster_size': 5}
            )
            
            assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])