from .contracts import Analyzer, AnalysisContext
from .models import Submission, AnalysisReport, Evidence

class LengthAnalyzer:
    analyzer_id = "length_analyzer"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        code_lines = [l for l in (submission.code or "").splitlines() if l.strip()]
        text_lines = [l for l in (submission.text or "").splitlines() if l.strip()]
        total_non_empty = len(code_lines) + len(text_lines)
        
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=None,
            rubric_feedback=f"Найдено {total_non_empty} непустых строк кода и текста."
        )

class TemplateAnalyzer:
    analyzer_id = "template_analyzer"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        template = context.assignment.allowed_template
        if not template:
            return AnalysisReport(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                similarity=0.0,
                rubric_feedback="Шаблон не задан."
            )

        content = (submission.code or "") + "\n" + (submission.text or "")
        lines = content.splitlines()
        matches = []
        
        template_lines = [l.strip() for l in template.splitlines() if l.strip()]
        
        for i, line in enumerate(lines, 1):
            if line.strip() in template_lines:
                matches.append(Evidence(line_number=i, text=line))

        found_unique_templates = set(e.text.strip() for e in matches)
        similarity = min(1.0, len(found_unique_templates) / max(len(template_lines), 1))
        
        feedback = f"Найдено {len(matches)} точных совпадений с разрешённым шаблоном. Совпадение разрешено."
        
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            similarity=similarity,
            rubric_feedback=feedback,
            evidence=matches
        )