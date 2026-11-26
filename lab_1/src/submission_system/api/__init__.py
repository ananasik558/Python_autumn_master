"""FastAPI-интерфейс анализатора ЛР3, этап Б."""

from .app import create_app
from .settings import Settings

__all__ = ["Settings", "create_app"]
