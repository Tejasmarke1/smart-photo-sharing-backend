from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.base import get_db

router = APIRouter()


@router.post("/signup")
async def signup(db: Session = Depends(get_db)):
    \"\"\"Register new photographer account.\"\"\"
    return {"message": "Signup endpoint - implementation pending"}


@router.post("/login")
async def login(db: Session = Depends(get_db)):
    \"\"\"Login and get JWT token.\"\"\"
    return {"message": "Login endpoint - implementation pending"}


@router.post("/refresh")
async def refresh_token():
    \"\"\"Refresh access token.\"\"\"
    return {"message": "Refresh endpoint - implementation pending"}
