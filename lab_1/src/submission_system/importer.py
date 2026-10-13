import json
from typing import List, Tuple
from .models import SubmissionResult, SubmissionError, ManifestEntry
from .validator import validate_submission
from .repository import InMemoryRepository
from .config import ProcessingConfig

class SubmissionImporter:
    def __init__(self, repo: InMemoryRepository, config: ProcessingConfig):
        self.repo = repo
        self.config = config
        self.manifest: List[ManifestEntry] = []

    def import_file(self, file_path: str) -> List[SubmissionResult]:
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    results.append(SubmissionResult(
                        status='error',
                        error=SubmissionError(code="FORMAT_ERROR", field=None, message="Неверный JSON", line_number=line_num)
                    ))
                    continue

                sub, err = validate_submission(data, line_num)
                if err:
                    results.append(SubmissionResult(status='error', error=err))
                    continue

                existing = self.repo.get(sub.submission_id)
                if existing:
                    if existing.raw_content_hash == sub.raw_content_hash:
                        results.append(SubmissionResult(status='duplicate', submission_id=sub.submission_id))
                    else:
                        results.append(SubmissionResult(
                            status='conflict', 
                            submission_id=sub.submission_id,
                            error=SubmissionError(code="CONTENT_CONFLICT", field="submission_id", message="ID совпадает, но содержимое другое")
                        ))
                else:
                    self.repo.save(sub)
                    self.manifest.append(ManifestEntry(
                        submission_id=sub.submission_id,
                        content_hash=sub.raw_content_hash,
                        processing_version=self.config.model_version
                    ))
                    results.append(SubmissionResult(status='accepted', submission_id=sub.submission_id))
        
        return results