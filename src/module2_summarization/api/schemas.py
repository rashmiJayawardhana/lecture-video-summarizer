from typing import Any, Optional

from pydantic import BaseModel


class Sentence(BaseModel):
    sentence: str
    timestamp_start: float
    timestamp_end: float


class ClassifiedSentence(BaseModel):
    sentence: str
    timestamp_start: float
    timestamp_end: float
    is_important: bool
    confidence: float
    keyword_boost: bool
    definition_match: bool
    repetition_boost: bool


class Segment(BaseModel):
    segment_id: str
    timestamp_start: float
    timestamp_end: float
    importance_ratio_T: float
    sentences: list[ClassifiedSentence]


class SemanticAnalysis(BaseModel):
    topic: str
    summary: str
    key_points: list[str] = []
    error: Optional[str] = None


class EnrichedBlock(BaseModel):
    block_id: str
    timestamp_start: float
    timestamp_end: float
    segments_count: int
    segment_ids: list[str]
    semantic_analysis: SemanticAnalysis


class ClassifyRequest(BaseModel):
    sentences: list[Sentence]


class EnrichRequest(BaseModel):
    segments: list[Segment]


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing" | "done" | "failed"
    step: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
