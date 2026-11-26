"""Совместимый фасад для исполнителя ЛР3.

Основная реализация находится в ``submission_system.jobs``. Модуль оставлен,
чтобы код и тесты могли импортировать исполнитель по короткому пути
``submission_system.executor``.
"""

from .jobs import *  # noqa: F401,F403
from .jobs import __all__ as _jobs_all

__all__ = list(_jobs_all)
