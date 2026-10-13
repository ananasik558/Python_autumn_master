import hashlib
from typing import List, Dict, Any
from .contracts import Analyzer, AnalysisContext, InputPreparer
from .models import Submission, Assignment, Evidence
from .sources import SubmissionSource

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