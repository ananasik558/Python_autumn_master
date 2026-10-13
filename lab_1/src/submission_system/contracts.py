from typing import Protocol, List, Dict, Any, Optional
from .models import Submission, AnalysisReport, Assignment
from dataclasses import dataclass, field

class Analyzer(Protocol):
    analyzer_id: str
    analyzer_version: str
    
    def analyze(self, submission: Submission, context: 'AnalysisContext') -> AnalysisReport: ...

class InputPreparer(Protocol):
    def prepare_input(self, submission: Submission) -> Dict[str, Any]: ...

class FeedbackFormatter(Protocol):
    def format_feedback(self, report: AnalysisReport) -> str: ...

@dataclass
class AnalysisContext:
    assignment: Assignment
    config_hash: str = ""