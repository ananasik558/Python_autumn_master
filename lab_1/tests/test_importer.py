import json
import pytest
from submission_system.models import SubmissionResult
from submission_system.repository import InMemoryRepository
from submission_system.importer import SubmissionImporter
from submission_system.validator import validate_submission
from submission_system.config import ProcessingConfig


@pytest.fixture
def config():
    """Фикстура конфигурации по умолчанию для большинства тестов."""
    return ProcessingConfig(mode='ml', model_version='1.0.0-baseline')


@pytest.fixture
def repo():
    return InMemoryRepository()


@pytest.fixture
def importer(repo, config):
    """Импортер с внедрённой конфигурацией."""
    return SubmissionImporter(repo, config)


def create_jsonl_file(tmp_path, lines: list[str]) -> str:
    file = tmp_path / "submissions.jsonl"
    with open(file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return str(file)


# 1. Успешно принимается корректная запись
def test_valid_import(tmp_path, importer, repo):
    valid_line = json.dumps({
        "student_id": "s1",
        "submission_id": "sub1",
        "timestamp": "2026-10-13T10:00:00+03:00",
        "code": "print('hello')"
    })
    file_path = create_jsonl_file(tmp_path, [valid_line])

    results = importer.import_file(file_path)

    assert len(results) == 1
    assert results[0].status == 'accepted'
    assert repo.get("sub1") is not None


# 2. Неверный timestamp дает объяснимую ошибку
def test_invalid_timestamp(tmp_path, importer):
    invalid_line = json.dumps({
        "student_id": "s1",
        "submission_id": "sub2",
        "timestamp": "not-a-date",
        "code": ""
    })
    file_path = create_jsonl_file(tmp_path, [invalid_line])

    results = importer.import_file(file_path)
    assert results[0].status == 'error'
    assert results[0].error.code == "INVALID_TIMESTAMP"


# 3. Пустой student_id дает объяснимую ошибку
def test_empty_student_id(tmp_path, importer):
    invalid_line = json.dumps({
        "student_id": "",
        "submission_id": "sub3",
        "timestamp": "2026-10-13T10:00:00+03:00",
        "code": ""
    })
    file_path = create_jsonl_file(tmp_path, [invalid_line])

    results = importer.import_file(file_path)
    assert results[0].status == 'error'
    assert results[0].error.code == "MISSING_FIELD"
    assert results[0].error.field == "student_id"


# 4. Дубликат не увеличивает число сдач
def test_duplicate_handling(tmp_path, importer, repo):
    content = {
        "student_id": "s1",
        "submission_id": "sub4",
        "timestamp": "2026-10-13T10:00:00+03:00",
        "code": "x=1"
    }
    lines = [json.dumps(content), json.dumps(content)]
    file_path = create_jsonl_file(tmp_path, lines)

    results = importer.import_file(file_path)

    assert results[0].status == 'accepted'
    assert results[1].status == 'duplicate'
    assert len(repo._storage) == 1


# 5. Конфликт не перезаписывает исходную запись
def test_conflict_handling(tmp_path, importer, repo):
    line1 = json.dumps({
        "student_id": "s1",
        "submission_id": "sub5",
        "timestamp": "2026-10-13T10:00:00+03:00",
        "code": "original"
    })
    line2 = json.dumps({
        "student_id": "s1",
        "submission_id": "sub5",
        "timestamp": "2026-10-13T11:00:00+03:00",
        "code": "modified"
    })
    file_path = create_jsonl_file(tmp_path, [line1, line2])

    results = importer.import_file(file_path)

    assert results[0].status == 'accepted'
    assert results[1].status == 'conflict'
    assert repo.get("sub5").code == "original"


# 6. Импорт повреждённой JSON-строки продолжает обработку остальных
def test_malformed_json_continues(tmp_path, importer):
    lines = [
        "{broken json",
        json.dumps({
            "student_id": "s1",
            "submission_id": "sub6",
            "timestamp": "2026-10-13T10:00:00+03:00",
            "code": "ok"
        })
    ]
    file_path = create_jsonl_file(tmp_path, lines)

    results = importer.import_file(file_path)

    assert len(results) == 2
    assert results[0].status == 'error'
    assert results[0].error.code == "FORMAT_ERROR"
    assert results[1].status == 'accepted'


# 7. Проверка номера строки в ошибке
def test_error_line_number(tmp_path, importer):
    lines = ["bad json", "also bad"]
    file_path = create_jsonl_file(tmp_path, lines)

    results = importer.import_file(file_path)

    assert results[0].error.line_number == 1
    assert results[1].error.line_number == 2


# 8. Тест на детерминированность конфигурации (AI/MLOps)
def test_config_deterministic():
    config_ml = ProcessingConfig(mode='ml')
    config_llm = ProcessingConfig(mode='llm')

    meta_ml = config_ml.get_metadata()
    meta_llm = config_llm.get_metadata()

    assert meta_ml['deterministic'] is True
    assert meta_llm['deterministic'] is True
    assert meta_ml['mode'] != meta_llm['mode']


# 9. Тест на заполнение манифеста (AI/MLOps)
def test_manifest_generation(tmp_path):
    repo = InMemoryRepository()
    config = ProcessingConfig(model_version='test-v1')
    importer = SubmissionImporter(repo, config)

    valid_line = '{"student_id": "s1", "submission_id": "sub_m1", "timestamp": "2026-10-13T10:00:00+03:00", "code": "x=1"}'
    file_path = create_jsonl_file(tmp_path, [valid_line])

    importer.import_file(file_path)

    assert len(importer.manifest) == 1
    assert importer.manifest[0].processing_version == 'test-v1'
    assert importer.manifest[0].content_hash is not None