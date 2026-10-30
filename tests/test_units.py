import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables.retry import RunnableRetry
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from chain import MAX_ATTEMPTS, PROMPT, build_chain, build_model, ensure_valid
from errors import RETRYABLE_ERRORS, InvalidOutputError, UnusableOutputError
from fake_api import GOOD_ENTITIES
from schemas import EntidadesTecnicas, NivelCriticidad

VALID = EntidadesTecnicas(**GOOD_ENTITIES)


def result_with(finish_key, finish_value, parsed=VALID, parsing_error=None):
    raw = AIMessage(content="", response_metadata={finish_key: finish_value})
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}


def test_esquema_acepta_el_ejemplo_de_la_guia():
    assert VALID.model_dump(mode="json") == GOOD_ENTITIES


@pytest.mark.parametrize("nivel", ["baja", "media", "alta"])
def test_criticidad_valida(nivel):
    assert EntidadesTecnicas(**{**GOOD_ENTITIES, "nivel_de_criticidad": nivel}).nivel_de_criticidad == NivelCriticidad(nivel)


def test_criticidad_fuera_del_enum():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**GOOD_ENTITIES, "nivel_de_criticidad": "critica"})


@pytest.mark.parametrize("tecnologias", [[], [""], ["  "], ["FastAPI", ""]])
def test_tecnologias_no_puede_ser_vacia_ni_tener_nombres_vacios(tecnologias):
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**GOOD_ENTITIES, "tecnologias": tecnologias})


def test_tecnologias_se_limpian_y_deduplican():
    entities = EntidadesTecnicas(**{**GOOD_ENTITIES, "tecnologias": [" Redis ", "Redis", "FastAPI"]})
    assert entities.tecnologias == ["Redis", "FastAPI"]


def test_resumen_vacio_es_invalido():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**GOOD_ENTITIES, "resumen_tecnico": "   "})


def test_ensure_valid_devuelve_el_objeto():
    assert ensure_valid(result_with("stop_reason", "end_turn")) is VALID


@pytest.mark.parametrize("key, value", [("stop_reason", "max_tokens"), ("finish_reason", "length")])
def test_ensure_valid_detecta_truncamiento_antes_de_transformar(key, value):
    with pytest.raises(UnusableOutputError, match="max_tokens"):
        ensure_valid(result_with(key, value, parsed=None, parsing_error=ValueError("json cortado")))


def test_ensure_valid_detecta_rechazo_del_modelo():
    with pytest.raises(UnusableOutputError, match="rechazo"):
        ensure_valid(result_with("stop_reason", "refusal", parsed=None))


def test_ensure_valid_convierte_el_error_de_parseo_en_error_reintentable():
    with pytest.raises(InvalidOutputError):
        ensure_valid(result_with("stop_reason", "end_turn", parsed=None, parsing_error=OutputParserException("x")))


def test_ensure_valid_sin_objeto_estructurado():
    with pytest.raises(InvalidOutputError):
        ensure_valid(result_with("stop_reason", "end_turn", parsed=None))


def test_el_prompt_pide_solo_el_texto_y_define_roles():
    assert PROMPT.input_variables == ["texto"]
    system, human = PROMPT.format_messages(texto="log de error")
    assert system.type == "system" and "Reglas de formato" in system.content
    assert human.type == "human" and human.content.endswith("log de error")


def test_la_cadena_usa_pipe_con_structured_output_y_with_retry():
    model = ChatOpenAI(model="gpt-4o-mini", api_key="k")
    chain = build_chain(model)
    assert isinstance(chain, RunnableRetry)
    assert chain.max_attempt_number == MAX_ATTEMPTS >= 2
    assert chain.retry_exception_types == RETRYABLE_ERRORS
    assert chain.bound.first == PROMPT
    assert chain.bound.last.func is ensure_valid


def test_los_errores_de_autenticacion_no_son_reintentables():
    import anthropic
    import openai

    assert not issubclass(anthropic.AuthenticationError, RETRYABLE_ERRORS)
    assert not issubclass(openai.AuthenticationError, RETRYABLE_ERRORS)
    assert not issubclass(openai.LengthFinishReasonError, RETRYABLE_ERRORS)


def test_build_model_anthropic_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    model = build_model()
    assert isinstance(model, ChatAnthropic) and model.max_retries == 0


def test_build_model_openai_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    model = build_model()
    assert isinstance(model, ChatOpenAI) and model.temperature == 0 and model.max_retries == 0


def test_build_model_sin_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_model("anthropic")


def test_build_model_proveedor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        build_model("gemini")
