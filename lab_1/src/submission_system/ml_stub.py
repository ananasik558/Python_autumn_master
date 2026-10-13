from .contracts import InputPreparer, FeedbackFormatter, Analyzer, AnalysisContext
from .models import Submission, AnalysisReport, Assignment

class StubModelAnalyzer:
    analyzer_id = "ml_stub"
    analyzer_version = "0.1.0-stub"
    feature_schema_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        content_len = len((submission.code or "") + (submission.text or ""))
        mock_score = min(content_len / 100.0, 1.0)
        
        feedback = f"[STUB] Модель оценила работу на {mock_score:.2f}. Schema признаков v{self.feature_schema_version}"
        
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=mock_score,
            rubric_feedback=feedback
        )

class SimpleInputPreparer(InputPreparer):
    def prepare_input(self, submission: Submission) -> dict:
        return {
            "code_length": len(submission.code or ""),
            "text_length": len(submission.text or ""),
            "has_ai_declaration": submission.ai_declaration is True
        }

class SimpleFeedbackFormatter(FeedbackFormatter):
    def format_feedback(self, report: AnalysisReport) -> str:
        parts = [f"Анализатор: {report.analyzer_id} v{report.analyzer_version}"]
        if report.score is not None:
            parts.append(f"Оценка: {report.score:.2f}")
        if report.rubric_feedback:
            parts.append(f"Комментарий: {report.rubric_feedback}")
        return " | ".join(parts)