import copy
import pytest
import json
from submission_system.models import Submission, Assignment
from submission_system.contracts import AnalysisContext, Analyzer
from submission_system.sources import InMemorySource, JsonlSource
from submission_system.analyzers import LengthAnalyzer, TemplateAnalyzer
from submission_system.analysis_service import AnalysisService
from submission_system.ml_stub import StubModelAnalyzer


@pytest.fixture
def sample_submissions():
    return [
        Submission(
            student_id="s1",
            submission_id="sub1",
            timestamp="2026-10-13T10:00:00+03:00",
            code="line1\nline2\n\nline3",
            text=""
        ),
        Submission(
            student_id="s2",
            submission_id="sub2",
            timestamp="2026-10-13T11:00:00+03:00",
            code="template_line\ntemplate_line",
            text=""
        ),
    ]


@pytest.fixture
def assignment():
    return Assignment(assignment_id="hw1", allowed_template="template_line")

def test_length_analyzer_in_memory(sample_submissions, assignment):
    source = InMemorySource(sample_submissions)
    service = AnalysisService([LengthAnalyzer()])
    results = service.run_analysis(source, assignment)

    assert len(results) == 2
    assert results[0]["analyses"]["length_analyzer"]["rubric_feedback"].startswith("Найдено 3")
    assert results[1]["analyses"]["length_analyzer"]["rubric_feedback"].startswith("Найдено 2")
    assert results[0]["analyses"]["length_analyzer"]["score"] is None


def test_template_analyzer_in_memory(sample_submissions, assignment):
    source = InMemorySource(sample_submissions)
    service = AnalysisService([TemplateAnalyzer()])
    results = service.run_analysis(source, assignment)

    analysis = results[1]["analyses"]["template_analyzer"]
    assert analysis["similarity"] == 1.0
    assert len(analysis["evidence"]) == 2
    assert "совпадение разрешено" in analysis["rubric_feedback"].lower()


def test_stub_model_deterministic(sample_submissions, assignment):
    source = InMemorySource(sample_submissions)
    service = AnalysisService([StubModelAnalyzer()])

    r1 = service.run_analysis(source, assignment)
    r2 = service.run_analysis(source, assignment)

    assert r1[0]["analyses"]["ml_stub"]["score"] == r2[0]["analyses"]["ml_stub"]["score"]
    assert "Schema признаков v1.0.0" in r1[0]["analyses"]["ml_stub"]["rubric_feedback"]


def test_multiple_analyzers_no_branching(sample_submissions, assignment):
    source = InMemorySource(sample_submissions)
    service = AnalysisService([LengthAnalyzer(), TemplateAnalyzer(), StubModelAnalyzer()])
    results = service.run_analysis(source, assignment)

    analyses = results[0]["analyses"]
    assert set(analyses.keys()) == {"length_analyzer", "template_analyzer", "ml_stub"}


def test_evidence_line_ranges(sample_submissions, assignment):
    source = InMemorySource(sample_submissions)
    service = AnalysisService([TemplateAnalyzer()])
    results = service.run_analysis(source, assignment)

    evidence_list = results[1]["analyses"]["template_analyzer"]["evidence"]
    content_lines_count = len((sample_submissions[1].code or "").splitlines())

    for ev in evidence_list:
        assert 1 <= ev["line_number"] <= content_lines_count, \
            f"Line number {ev['line_number']} out of range [1, {content_lines_count}]"


def test_submission_immutability_during_analysis(sample_submissions, assignment):
    original_code = sample_submissions[0].code
    original_text = sample_submissions[0].text
    original_ai_decl = sample_submissions[0].ai_declaration

    service = AnalysisService([LengthAnalyzer(), TemplateAnalyzer()])
    source = InMemorySource(sample_submissions)
    service.run_analysis(source, assignment)

    assert sample_submissions[0].code == original_code
    assert sample_submissions[0].text == original_text
    assert sample_submissions[0].ai_declaration == original_ai_decl


def test_jsonl_source_resource_release(tmp_path):
    file = tmp_path / "test.jsonl"
    record = {
        "student_id": "s1",
        "submission_id": "sub_res",
        "timestamp": "2026-10-13T10:00:00+03:00",
        "code": "x=1"
    }
    file.write_text(json.dumps(record) + "\n")

    source = JsonlSource(str(file))
    subs = list(source.get_submissions())

    assert len(subs) == 1
    assert subs[0].submission_id == "sub_res"
    
    with open(file, 'r') as f:
        content = f.read()
        assert "sub_res" in content


class BrokenAnalyzer:
    analyzer_id = "broken_analyzer"
    analyzer_version = "0.0.0-error"

    def analyze(self, submission: Submission, context: AnalysisContext):
        raise RuntimeError("Simulated analyzer failure")


def test_service_handles_analyzer_error(sample_submissions, assignment):
    service = AnalysisService([LengthAnalyzer(), BrokenAnalyzer()])
    source = InMemorySource(sample_submissions)
    results = service.run_analysis(source, assignment)

    assert "length_analyzer" in results[0]["analyses"]
    assert "rubric_feedback" in results[0]["analyses"]["length_analyzer"]

    broken_result = results[0]["analyses"]["broken_analyzer"]
    assert "error" in broken_result
    assert "Simulated analyzer failure" in broken_result["error"]


def test_contract_tests_work_with_both_sources(sample_submissions, assignment, tmp_path):
    file = tmp_path / "contract_test.jsonl"
    lines = []
    for sub in sample_submissions:
        obj = {
            "student_id": sub.student_id,
            "submission_id": sub.submission_id,
            "timestamp": sub.timestamp,
            "code": sub.code,
            "text": sub.text
        }
        lines.append(json.dumps(obj))
        
    file.write_text("\n".join(lines) + "\n")

    sources = [
        ("in-memory", InMemorySource(sample_submissions)),
        ("jsonl", JsonlSource(str(file))),
    ]

    for source_name, source in sources:
        service = AnalysisService([LengthAnalyzer()])
        results = service.run_analysis(source, assignment)

        if len(results) != 2:
            print(f"DEBUG [{source_name}]: Got {len(results)} results instead of 2")
            if source_name == "jsonl":
                with open(file, 'r') as f:
                    print(f"DEBUG File content: {f.read()[:200]}...")

        assert len(results) == 2, f"Failed for {source_name}: expected 2, got {len(results)}"
        
        analysis = results[0]["analyses"]["length_analyzer"]
        assert isinstance(analysis, dict), f"Expected dict, got {type(analysis)} for {source_name}"
        assert analysis["rubric_feedback"].startswith("Найдено 3"), \
            f"Failed for {source_name}: {analysis}"