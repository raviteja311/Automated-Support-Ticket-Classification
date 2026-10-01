from pydantic import BaseModel, Field

# A support message is a few sentences. The cap stops one request from
# pinning a worker on a multi-megabyte body, and keeps the prediction log sane.
MAX_TEXT_LENGTH = 5000


class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        examples=["I was charged twice this month"],
    )


class TicketResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]
