import asyncio
import logging

from dotenv import load_dotenv

from chain import build_chain, build_model, process_text
from errors import ExtractionFailed

SAMPLES = {
    "Descripción de arquitectura": (
        "Nuestra API en FastAPI usa Redis como caché y PostgreSQL para persistencia. "
        "En horas pico se agotan las conexiones concurrentes a la base de datos y los "
        "usuarios reciben errores 500."
    ),
    "Log de error": (
        "2026-10-29 14:02:11 ERROR [worker-3] kafka.consumer: CommitFailedException: "
        "group rebalance in progress. Mensajes duplicados en el servicio de pagos."
    ),
    "Texto ambiguo (prueba de estrés)": (
        "A veces todo va lento, creo que es por la base de datos o algo así, no estoy seguro."
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


async def main() -> None:
    load_dotenv()
    configure_logging()
    try:
        chain = build_chain(build_model())
    except ValueError as exc:
        print(f"Configuracion invalida: {exc}")
        return

    for label, text in SAMPLES.items():
        print(f"\n=== {label} ===")
        try:
            entities = await process_text(text, chain)
        except ExtractionFailed as exc:
            print(f"No se pudo extraer: {exc}")
            continue
        print(entities.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
