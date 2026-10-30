import anthropic
import openai
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

TRUNCATION_REASONS = frozenset({"max_tokens", "length"})
REFUSAL_REASONS = frozenset({"refusal"})


class InvalidOutputError(Exception): ...


class UnusableOutputError(Exception): ...


class ExtractionFailed(Exception): ...


FORMAT_ERRORS = (InvalidOutputError, OutputParserException, ValidationError)

TRANSIENT_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

RETRYABLE_ERRORS = FORMAT_ERRORS + TRANSIENT_ERRORS
