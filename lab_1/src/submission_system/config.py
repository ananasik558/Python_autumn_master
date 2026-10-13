from dataclasses import dataclass
from typing import Literal

@dataclass
class ProcessingConfig:
    mode: Literal['ml', 'llm'] = 'ml'
    model_version: str = '1.0.0-baseline'
    
    def get_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "version": self.model_version,
            "deterministic": True
        }