import json
import logging
from typing import Iterator, List, Protocol
from .models import Submission

logger = logging.getLogger(__name__)

class SubmissionSource(Protocol):
    def get_submissions(self) -> Iterator[Submission]: ...

class JsonlSource:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_submissions(self) -> Iterator[Submission]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                
                    safe_data = {
                        "student_id": data.get("student_id"),
                        "submission_id": data.get("submission_id"),
                        "timestamp": data.get("timestamp"),
                        "code": data.get("code", ""),
                        "text": data.get("text", ""),
                        "ai_declaration": data.get("ai_declaration"),
                        "raw_content_hash": data.get("raw_content_hash")
                    }
                    
                    yield Submission(**safe_data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Failed to create Submission at line {line_num}: {type(e).__name__}: {e}")

class InMemorySource:
    def __init__(self, submissions: List[Submission]):
        self.submissions = submissions

    def get_submissions(self) -> Iterator[Submission]:
        return iter(self.submissions)