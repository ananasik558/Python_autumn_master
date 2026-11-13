import hashlib
import math
from typing import List, Dict, Any
from .contracts import Analyzer, AnalysisContext, InputPreparer
from .models import Submission, Assignment, Evidence, AnalysisReport
from .sources import SubmissionSource


def validate_analysis_report(report: AnalysisReport) -> None:
    """Отклоняет невалидный результат AI/ML-анализатора.

    Модель может вернуть None, NaN, бесконечность или score вне [0, 1].
    Такой результат нельзя молча записывать как оценку: сервис помечает
    анализ как failed, а не интерпретирует мусор как «нулевое сходство».
    """

    if not isinstance(report, AnalysisReport):
        raise TypeError(
            f"analyzer returned {type(report).__name__}, expected AnalysisReport"
        )
    for field_name in ("score", "similarity"):
        value = getattr(report, field_name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} is not numeric: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name} is not finite: {number!r}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{field_name} is outside [0, 1]: {number!r}")


class AnalysisService:
    def __init__(self, analyzers: List[Analyzer], input_preparer: InputPreparer | None = None):
        self.analyzers = {a.analyzer_id: a for a in analyzers}
        self.input_preparer = input_preparer

    def run_analysis(self, source: SubmissionSource, assignment: Assignment) -> List[Dict[str, Any]]:
        results = []
        context = AnalysisContext(
            assignment=assignment,
            config_hash=self._calculate_config_hash()
        )
        
        for submission in source.get_submissions():
            submission_results: Dict[str, Any] = {}
            
            for analyzer_id, analyzer in self.analyzers.items():
                try:
                    report = analyzer.analyze(submission, context)
                    validate_analysis_report(report)
                    
                    submission_results[analyzer_id] = {
                        "score": report.score,
                        "rubric_feedback": report.rubric_feedback,
                        "similarity": report.similarity,
                        "evidence": [
                            {"line_number": e.line_number, "text": e.text, "type": e.type}
                            for e in report.evidence
                        ]
                    }
                except Exception as e:
                    submission_results[analyzer_id] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            results.append({
                "submission_id": submission.submission_id,
                "analyses": submission_results
            })
            
        return results

    def _calculate_config_hash(self) -> str:
        config_data = "".join(f"{aid}:{av}" for aid, av in 
                            [(a.analyzer_id, a.analyzer_version) for a in self.analyzers.values()])
        return hashlib.sha256(config_data.encode()).hexdigest()[:16]