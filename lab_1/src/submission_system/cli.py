import sys
import json
from .repository import InMemoryRepository
from .importer import SubmissionImporter
from .config import ProcessingConfig

def main():
    if len(sys.argv) < 2:
        print("Использование: python -m submission_system.cli <path_to_jsonl>")
        sys.exit(1)

    file_path = sys.argv[1]
    config = ProcessingConfig(mode='ml') # По умолчанию ML
    repo = InMemoryRepository()
    importer = SubmissionImporter(repo, config)
    
    try:
        results = importer.import_file(file_path)
        output = {
            "summary": {
                "total": len(results),
                "accepted": sum(1 for r in results if r.status == 'accepted'),
                "duplicates": sum(1 for r in results if r.status == 'duplicate'),
                "conflicts": sum(1 for r in results if r.status == 'conflict'),
                "errors": sum(1 for r in results if r.status == 'error')
            },
            "manifest": [
                {
                    "submission_id": m.submission_id,
                    "hash": m.content_hash,
                    "version": m.processing_version
                } for m in importer.manifest
            ],
            "details": [
                {
                    "status": r.status,
                    "submission_id": r.submission_id,
                    "error": r.error.__dict__ if r.error else None
                } for r in results
            ]
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print(json.dumps({"error": "Файл не найден"}))
        sys.exit(1)

if __name__ == "__main__":
    main()