"""Акторы и проверки доступа ЛР3, этап Б.

Роли:
* ``student`` — видит и запускает только свои сдачи;
* ``teacher`` — видит отчёты только по своим заданиям, запуск/отмена запрещены;
* ``admin`` — полный доступ (локальный/служебный ключ).

Ключи сравниваются через ``secrets.compare_digest``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Literal, Optional

from ..models import Submission
from .settings import Settings


@dataclass(frozen=True)
class Actor:
    actor_id: str
    role: Literal["admin", "student", "teacher"]
    student_id: Optional[str] = None
    assignment_ids: frozenset[str] = frozenset()

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_student(self) -> bool:
        return self.role == "student"

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"

    def can_read_submission(self, submission: Submission) -> bool:
        if self.is_admin:
            return True
        if self.is_student:
            return bool(self.student_id) and submission.student_id == self.student_id
        if self.is_teacher:
            return submission.assignment_id in self.assignment_ids
        return False

    def can_start_submission(self, submission: Submission) -> bool:
        if self.is_admin:
            return True
        if self.is_student:
            return bool(self.student_id) and submission.student_id == self.student_id
        return False


def resolve_actor(settings: Settings, provided_key: str | None) -> Optional[Actor]:
    if not provided_key:
        return None
    if settings.api_key and secrets.compare_digest(provided_key, settings.api_key):
        return Actor(actor_id="admin", role="admin")
    for key, student_id in settings.student_keys.items():
        if secrets.compare_digest(provided_key, str(key)):
            return Actor(
                actor_id=str(student_id),
                role="student",
                student_id=str(student_id),
            )
    for key, assignment_ids in settings.teacher_keys.items():
        if secrets.compare_digest(provided_key, str(key)):
            return Actor(
                actor_id=str(key),
                role="teacher",
                assignment_ids=frozenset(str(item) for item in assignment_ids),
            )
    return None
