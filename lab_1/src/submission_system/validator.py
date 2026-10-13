import hashlib
import json
from typing import Optional
from datetime import datetime
from .models import Submission, SubmissionError

def calculate_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def validate_submission(data: dict, line_num: int) -> tuple[Optional[Submission], Optional[SubmissionError]]:
    required_fields = ['student_id', 'submission_id', 'timestamp']
    
    for f in required_fields:
        if f not in data or not str(data[f]).strip():
            return None, SubmissionError(
                code="MISSING_FIELD", 
                field=f, 
                message=f"Отсутствует или пусто обязательное поле: {f}",
                line_number=line_num
            )

    try:
        ts = data['timestamp']
        if 'T' not in ts: raise ValueError
        datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None, SubmissionError(
            code="INVALID_TIMESTAMP", 
            field="timestamp", 
            message="Неверный формат времени (ожидается ISO 8601)",
            line_number=line_num
        )

    raw_code = data.get('code', '') or ''
    raw_text = data.get('text', '') or ''
    combined_content = raw_code + raw_text
    
    sub = Submission(
        student_id=str(data['student_id']),
        submission_id=str(data['submission_id']),
        timestamp=data['timestamp'],
        code=raw_code,
        text=raw_text,
        ai_declaration=data.get('ai_declaration'),
        raw_content_hash=calculate_hash(combined_content)
    )
    
    return sub, None