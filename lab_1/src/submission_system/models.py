from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class SubmissionError:
    code: str
    field: Optional[str]
    message: str
    line_number: Optional[int] = None

@dataclass
class SubmissionResult:
    status: str  # 'accepted', 'duplicate', 'conflict'
    submission_id: Optional[str] = None
    error: Optional[SubmissionError] = None

@dataclass
class Submission:
    student_id: str
    submission_id: str
    timestamp: str
    code: Optional[str] = None
    text: Optional[str] = None
    ai_declaration: Optional[bool] = None
    
    raw_content_hash: Optional[str] = None 

@dataclass
class ManifestEntry:
    submission_id: str
    content_hash: str
    processing_version: str
    timestamp_processed: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AnalysisReport:
    analyzer_id: str
    analyzer_version: str
    score: Optional[float] = None
    rubric_feedback: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)