from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Concept(BaseModel):
    id: int = Field(..., ge=1, le=5)
    term: str = Field(..., description="Core concept or axiomatic term")
    definition: str = Field(..., description="Strict definition grounded only in source text")
    source_evidence: str = Field(..., description="Exact quote or paraphrase from input")


class ConceptExtractionOutput(BaseModel):
    concepts: List[Concept] = Field(..., min_length=5, max_length=5)

    @field_validator("concepts", mode="before")
    @classmethod
    def trim_concepts(cls, v):
        if isinstance(v, list) and len(v) > 5:
            return v[:5]
        return v


class MCQItem(BaseModel):
    concept_id: int
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_option_index: int = Field(..., ge=0, le=3)

    @field_validator("options", mode="before")
    @classmethod
    def trim_options(cls, v):
        if isinstance(v, list) and len(v) > 4:
            return v[:4]
        return v


class FlashcardItem(BaseModel):
    concept_id: int
    front: str
    back: str


class QuestionGenerationOutput(BaseModel):
    mcqs: List[MCQItem] = Field(..., min_length=5, max_length=5)
    flashcards: List[FlashcardItem] = Field(..., min_length=3, max_length=3)

    @field_validator("mcqs", mode="before")
    @classmethod
    def trim_mcqs(cls, v):
        if isinstance(v, list) and len(v) > 5:
            return v[:5]
        return v

    @field_validator("flashcards", mode="before")
    @classmethod
    def trim_flashcards(cls, v):
        if isinstance(v, list) and len(v) > 3:
            return v[:3]
        return v


class ReviewedMCQ(BaseModel):
    question: str
    options: List[str]
    correct_option: str
    distractor_justifications: List[str] = Field(
        ..., description="Short explanation of why each of the 3 incorrect options is wrong"
    )
    is_valid: bool
    issue_note: str = ""


class PedagogicalReviewOutput(BaseModel):
    status: Literal["APPROVED", "PEDAGOGICAL_FAILURE"]
    failure_reason: str = Field(
        default="",
        description="Explicit alert if source material was insufficient, circular, or caused hallucinations",
    )
    reviewed_mcqs: List[ReviewedMCQ] = Field(default_factory=list)
    approved_flashcards: List[FlashcardItem] = Field(default_factory=list)