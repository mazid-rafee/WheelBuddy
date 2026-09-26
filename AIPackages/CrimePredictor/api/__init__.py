"""Local FastAPI inference service for CrimeRateMLP."""

from .app import app, create_app

__all__ = ["app", "create_app"]
