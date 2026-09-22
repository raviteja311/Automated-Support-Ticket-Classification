from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["I was charged twice this month"])


class TicketResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]
