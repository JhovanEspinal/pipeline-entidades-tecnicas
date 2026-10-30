# Pipeline de extracción de entidades técnicas

Pre-Entrega 2 (Módulo 2, AI Engineering). Recibe un párrafo sin procesar (descripción de arquitectura o log de error) y devuelve un **objeto Pydantic validado**, con una cadena LCEL asíncrona y reintentos.

```
{"texto": "..."} → ChatPromptTemplate → model.with_structured_output(EntidadesTecnicas) → validación → EntidadesTecnicas
                                         └──────────────── .with_retry() envuelve toda la cadena ────────────────┘
```

## Estructura

| Archivo | Rol |
|---|---|
| `schemas.py` | Modelo Pydantic `EntidadesTecnicas` y enum `NivelCriticidad` |
| `chain.py` | Prompt, modelo (`ChatAnthropic` / `ChatOpenAI`), cadena LCEL con `.with_structured_output()` y `.with_retry()`, y `process_text()` |
| `errors.py` | Excepciones del pipeline y qué errores se reintentan |
| `observability.py` | Logs de intentos y de errores del modelo |
| `main.py` | Mini-script asíncrono de prueba (3 textos, incluido uno ambiguo) |
| `tests/` | Tests sin API keys ni red externa |

## Salida esperada

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

Restricciones del contrato (`schemas.py`): `tecnologias` no vacía, sin nombres en blanco y sin duplicados; `nivel_de_criticidad` ∈ {`baja`, `media`, `alta`}; `resumen_tecnico` no vacío.

## Instalación y ejecución

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # luego completa tu API key
python main.py                  # ejemplo + prueba de estrés
python -m pytest                # tests
```

### Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `anthropic` u `openai` (por defecto `anthropic`) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Clave del proveedor elegido |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Opcionales. Por defecto `claude-haiku-5-5` / `gpt-4o-mini` |

### Uso desde código

```python
from chain import process_text

entidades = await process_text("Nuestra API en FastAPI usa Redis y PostgreSQL...")
print(entidades.model_dump_json(indent=2))
```

## Cómo cumple el entregable

| Requisito | Dónde |
|---|---|
| Esquema Pydantic | `schemas.py` |
| Prompt modular con texto + instrucciones de formato (sin f-strings) | `PROMPT` en `chain.py`: variables `{texto}` e `{instrucciones_formato}` (esta última con `.partial()`) |
| Cadena LCEL `prompt \| model.with_structured_output(Schema)` | `build_chain()` |
| Reintento con `.with_retry()` ante JSON mal formado o incompleto | `build_chain()` + `errors.RETRYABLE_ERRORS` |
| `process_text()` asíncrona con `.ainvoke()` y logs | `chain.py` |
| Mini-script de prueba | `main.py` |

## Resiliencia: qué se reintenta y qué no

Siguiendo el criterio del curso (reintentar solo lo que puede ser transitorio):

| Situación | ¿Reintenta? |
|---|---|
| JSON mal formado, campo faltante, `tecnologias` vacía, valor fuera del enum | Sí (el modelo puede acertar en otro intento) |
| Rate limit (429), error de red, timeout, error 5xx | Sí, con backoff exponencial con jitter |
| API key inválida (401/403) | No |
| Respuesta **truncada** (`finish_reason` = `max_tokens` / `length`) | No: se aborta con un mensaje claro |
| Rechazo del modelo (`refusal`) | No |

**`finish_reason`:** el modelo se envuelve con `with_structured_output(..., include_raw=True)` para conservar el mensaje original. `ensure_valid()` lee `stop_reason` / `finish_reason` **antes** de transformar el resultado: si la respuesta se cortó por límite de tokens, falla de inmediato con `UnusableOutputError` en lugar de reintentar (volver a pedir lo mismo con el mismo límite solo gasta dinero). En OpenAI, el SDK ya lanza `LengthFinishReasonError` en ese caso y tampoco se reintenta.

Los intentos son 3 en total (1 inicial + 2 reintentos, `MAX_ATTEMPTS`). Los SDKs se crean con `max_retries=0` para que `.with_retry()` sea la única política de reintentos. Si se agotan, `process_text()` lanza `ExtractionFailed` (con la causa original encadenada); `main.py` lo captura, informa y continúa con el siguiente texto.

## Logs

Ejemplo (respuesta simulada con un 429 y dos salidas inválidas antes de recuperarse):

```
20:10:49 | INFO    | Procesando texto de 182 caracteres
20:10:49 | INFO    | Llamando al modelo (intento 1)
20:10:49 | WARNING | Error del modelo (intento 1): AnthropicRateLimitError: Error code: 429 ...
20:10:50 | INFO    | Llamando al modelo (intento 2)
20:10:51 | INFO    | Validacion OK: 3 tecnologias, criticidad=alta
20:10:51 | INFO    | Llamando al modelo (intento 1)
20:10:51 | WARNING | Validacion fallida: Failed to parse EntidadesTecnicas ... resumen_tecnico | Field required ...
20:10:53 | INFO    | Llamando al modelo (intento 2)
20:10:53 | INFO    | Validacion OK: 3 tecnologias, criticidad=alta
```

No se registra el contenido del texto analizado, solo su longitud.

## Notas de diseño

- **Salida estructurada por proveedor:** `ChatAnthropic` usa `method="json_schema"` (la función de salidas estructuradas de Anthropic, que no depende de forzar una herramienta); `ChatOpenAI` usa su método por defecto. Ambos pasan por la misma cadena.
- **Temperatura:** la API actual de Anthropic rechaza `temperature`, así que `ChatAnthropic` no la recibe. `ChatOpenAI` usa `temperature=0` por ser extracción.
- **Texto ambiguo:** si el modelo no encuentra tecnologías, `tecnologias` vacía viola el contrato y se reintenta; si tras los intentos sigue fallando, el error es explícito. Pydantic valida la **forma**, no que el contenido sea cierto.
- **Prompt injection:** el prompt indica que el texto es material a analizar y no instrucciones (defensa básica, no absoluta).

## Pruebas

`python -m pytest` (53 tests): esquema, `ensure_valid`, y la cadena completa con `ChatAnthropic` y `ChatOpenAI` **reales** contra un servidor HTTP local que simula las APIs (JSON mal formado, campos faltantes, truncamiento, 429, 500, 401). Aún no se ha ejecutado contra los servidores reales de OpenAI/Anthropic: para eso se necesita una API key en `.env`.
