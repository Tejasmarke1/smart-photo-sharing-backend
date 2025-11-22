# src/app/dependencies.py
from fastapi import Depends
from .config import get_settings

def get_app_settings():
    return get_settings()
