import json

import pytest

import main
from fake_api import GOOD_ENTITIES, FakeAPI, http_failure


@pytest.fixture
def api(monkeypatch):
    server = FakeAPI()
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    yield server
    server.close()


async def test_main_procesa_los_tres_ejemplos(api, capsys):
    await main.main()
    out = capsys.readouterr().out
    assert out.count('"nivel_de_criticidad": "alta"') == len(main.SAMPLES)
    assert json.dumps(GOOD_ENTITIES["tecnologias"][0]) in out
    assert len(api.requests) == len(main.SAMPLES)


async def test_main_informa_el_fallo_y_continua_con_el_siguiente(api, capsys):
    api.script = [http_failure(401)]
    await main.main()
    out = capsys.readouterr().out
    assert out.count("No se pudo extraer") == 1
    assert out.count('"nivel_de_criticidad"') == len(main.SAMPLES) - 1


async def test_main_con_configuracion_invalida_no_se_cae(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    await main.main()
    assert "Configuracion invalida" in capsys.readouterr().out
