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


NEEDS_REVIEW = "needs_review"


class TicketResponse(BaseModel):
    # A queue name, or NEEDS_REVIEW when the model is not confident enough to
    # route the message on its own.
    label: str
    # The model's top queue, always present, so a person reviewing the
    # message starts from the model's suggestion.
    predicted_label: str
    confidence: float
    all_scores: dict[str, float]
