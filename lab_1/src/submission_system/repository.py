from typing import Dict, Optional
from .models import Submission

class InMemoryRepository:
    def __init__(self):
        self._storage: Dict[str, Submission] = {}

    def get(self, submission_id: str) -> Optional[Submission]:
        return self._storage.get(submission_id)

    def save(self, submission: Submission) -> bool:
        if submission.submission_id in self._storage:
            return False
        self._storage[submission.submission_id] = submission
        return True