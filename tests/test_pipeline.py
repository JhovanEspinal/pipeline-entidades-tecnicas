import logging

import anthropic
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from chain import build_chain, process_text
from errors import ExtractionFailed
from fake_api import (
    EMPTY_LIST_TEXT,
    GOOD_ENTITIES,
    MALFORMED_TEXT,
    MISSING_FIELD_TEXT,
    FakeAPI,
    completion,
    http_failure,
)
from schemas import EntidadesTecnicas, NivelCriticidad

TEXT = "Nuestra API en FastAPI usa Redis y PostgreSQL y se agotan las conexiones."
EXPECTED = EntidadesTecnicas(**GOOD_ENTITIES)


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()


def anthropic_model(api):
    return ChatAnthropic(
        model="claude-haiku-5-5", max_tokens=300, max_retries=0, api_key="k", base_url=api.url
    )


def openai_model(api):
    return ChatOpenAI(
        model="gpt-4o-mini", max_tokens=300, max_retries=0, api_key="k", base_url=f"{api.url}/v1"
    )


@pytest.fixture(params=[anthropic_model, openai_model], ids=["anthropic", "openai"])
def chain(request, api):
    return build_chain(request.param(api), wait_exponential_jitter=False)


async def test_devuelve_objeto_validado(chain, api):
    result = await process_text(TEXT, chain)
    assert result == EXPECTED
    assert result.nivel_de_criticidad is NivelCriticidad.ALTA
    assert len(api.requests) == 1


async def test_json_mal_formado_se_recupera_con_un_reintento(chain, api, caplog):
    api.script = [completion(MALFORMED_TEXT)]
    with caplog.at_level(logging.INFO):
        result = await process_text(TEXT, chain)
    assert result == EXPECTED
    assert len(api.requests) == 2
    assert "Llamando al modelo (intento 2)" in caplog.text


async def test_campo_faltante_se_recupera(chain, api):
    api.script = [completion(MISSING_FIELD_TEXT)]
    assert await process_text(TEXT, chain) == EXPECTED
    assert len(api.requests) == 2


async def test_lista_de_tecnologias_vacia_se_recupera(chain, api):
    api.script = [completion(EMPTY_LIST_TEXT)]
    assert await process_text(TEXT, chain) == EXPECTED
    assert len(api.requests) == 2


async def test_formato_invalido_persistente_agota_los_intentos(chain, api):
    api.script = [completion(MALFORMED_TEXT)] * 5
    with pytest.raises(ExtractionFailed):
        await process_text(TEXT, chain)
    assert len(api.requests) == 3


async def test_respuesta_truncada_no_se_reintenta(chain, api):
    api.script = [completion(MALFORMED_TEXT, truncated=True)] * 5
    with pytest.raises(ExtractionFailed):
        await process_text(TEXT, chain)
    assert len(api.requests) == 1


async def test_rate_limit_se_reintenta_y_se_recupera(chain, api):
    api.script = [http_failure(429), http_failure(429)]
    assert await process_text(TEXT, chain) == EXPECTED
    assert len(api.requests) == 3


async def test_rate_limit_persistente_agota_los_intentos(chain, api):
    api.script = [http_failure(429)] * 5
    with pytest.raises(ExtractionFailed, match="RateLimitError"):
        await process_text(TEXT, chain)
    assert len(api.requests) == 3


async def test_error_de_servidor_se_reintenta(chain, api):
    api.script = [http_failure(500)]
    assert await process_text(TEXT, chain) == EXPECTED
    assert len(api.requests) == 2


async def test_api_key_invalida_no_se_reintenta(chain, api):
    api.script = [http_failure(401)] * 5
    with pytest.raises(ExtractionFailed) as failure:
        await process_text(TEXT, chain)
    assert isinstance(
        failure.value.__cause__, (anthropic.AuthenticationError, openai.AuthenticationError)
    )
    assert len(api.requests) == 1


async def test_los_logs_registran_la_validacion(chain, caplog):
    with caplog.at_level(logging.INFO):
        await process_text(TEXT, chain)
    assert "Procesando texto de" in caplog.text
    assert "Validacion OK: 3 tecnologias, criticidad=alta" in caplog.text


async def test_los_logs_registran_cada_intento_y_el_error_final(chain, api, caplog):
    api.script = [completion(MISSING_FIELD_TEXT)] * 5
    with caplog.at_level(logging.INFO), pytest.raises(ExtractionFailed):
        await process_text(TEXT, chain)
    assert caplog.text.count("Llamando al modelo") == 3
    assert caplog.text.count("Field required") >= 3
    assert "Extraccion fallida" in caplog.text


async def test_anthropic_envia_el_esquema_con_salida_estructurada(api):
    await process_text(TEXT, build_chain(anthropic_model(api), wait_exponential_jitter=False))
    sent = api.requests[0]["body"]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["messages"][0]["content"].endswith(TEXT)
    assert "Reglas de formato" in sent["system"]
    assert "temperature" not in sent


async def test_openai_envia_el_esquema_con_salida_estructurada(api):
    await process_text(TEXT, build_chain(openai_model(api), wait_exponential_jitter=False))
    sent = api.requests[0]["body"]
    assert sent["response_format"]["type"] == "json_schema"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "Reglas de formato" in sent["messages"][0]["content"]
