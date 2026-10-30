from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Entidades técnicas extraídas de un texto de arquitectura o de un log de error."
        }
    )

    tecnologias: list[str] = Field(
        min_length=1,
        description="Tecnologías, frameworks, servicios o herramientas mencionadas en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Criticidad del problema o riesgo descrito: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Resumen técnico de una o dos oraciones.",
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_sin_vacios_ni_duplicados(cls, tecnologias: list[str]) -> list[str]:
        cleaned = [name.strip() for name in tecnologias]
        if not all(cleaned):
            raise ValueError("tecnologias no puede contener nombres vacios")
        return list(dict.fromkeys(cleaned))

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_sin_espacios_sobrantes(cls, summary: str) -> str:
        stripped = summary.strip()
        if not stripped:
            raise ValueError("resumen_tecnico no puede estar vacio")
        return stripped
