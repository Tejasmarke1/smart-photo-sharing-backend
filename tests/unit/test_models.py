import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.models import User, Album, Photo, UserRole


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_user(db_session):
    """Test user creation."""
    user = User(
        name="Test User",
        email="test@example.com",
        phone="+919876543210",
        hashed_password="hashed_password",
        role=UserRole.PHOTOGRAPHER,
    )
    db_session.add(user)
    db_session.commit()
    
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.role == UserRole.PHOTOGRAPHER


def test_create_album(db_session):
    """Test album creation."""
    user = User(
        name="Photographer",
        email="photo@example.com",
        phone="+919876543210",
        hashed_password="hashed",
        role=UserRole.PHOTOGRAPHER,
    )
    db_session.add(user)
    db_session.commit()
    
    album = Album(
        photographer_id=user.id,
        title="Wedding Album",
        description="Beautiful wedding",
    )
    db_session.add(album)
    db_session.commit()
    
    assert album.id is not None
    assert album.photographer_id == user.id
    assert album.sharing_code is not None