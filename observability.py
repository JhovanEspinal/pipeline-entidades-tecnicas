import logging

from langchain_core.callbacks import AsyncCallbackHandler

log = logging.getLogger("pipeline")


NOISE_LINE_PREFIXES = ("For further information", "For troubleshooting")
MAX_LOGGED_ERROR_CHARS = 400


def one_line(error: object) -> str:
    meaningful_lines = [
        line.strip()
        for line in str(error).splitlines()
        if line.strip() and not line.strip().startswith(NOISE_LINE_PREFIXES)
    ]
    text = " | ".join(meaningful_lines)
    if len(text) <= MAX_LOGGED_ERROR_CHARS:
        return text
    return text[: MAX_LOGGED_ERROR_CHARS - 3] + "..."


class PipelineLogger(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.attempt = 0

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.attempt += 1
        log.info("Llamando al modelo (intento %d)", self.attempt)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        log.warning(
            "Error del modelo (intento %d): %s: %s",
            self.attempt, type(error).__name__, one_line(error),
        )
