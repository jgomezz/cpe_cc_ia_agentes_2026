"""
Lab 2.1 — MCP Server con FastMCP (Institución Educativa)
──────────────────────────────────────────

"""

import json
import unicodedata
from fastmcp import FastMCP

# Crear el servidor MCP
mcp = FastMCP("Institución Educativa Asistente Académico")


# ═══════════════════════════════════════════════════════════
# HELPER: normalizar texto (quitar tildes + lowercase)
# ═══════════════════════════════════════════════════════════

def normalizar(texto: str) -> str:
    """Convierte 'Cálculo' → 'calculo', 'Programación' → 'programacion'."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return sin_tildes.lower().strip()

# ═══════════════════════════════════════════════════════════
# DATOS SIMULADOS 
# ═══════════════════════════════════════════════════════════
ESTUDIANTES = {
    "T20231": {
        "nombre":     "María Quispe Gomez",
        "carrera":    "Desarrollo de Software",
        "ciclo":      3,
        "promedio":   17.0,
        "asistencia": "95%",
        "estado":     "Regular",
    },
    "T20232": {
        "nombre":     "Carlos Mendoza",
        "carrera":    "Big Data y Ciencia de Datos",
        "ciclo":      4,
        "promedio":   14.6,
        "asistencia": "70%",
        "estado":     "Alerta de asistencia",
    },
    "T20233": {
        "nombre":     "Lucía Vargas",
        "carrera":    "Redes y Comunicaciones",
        "ciclo":      2,
        "promedio":   11.0,
        "asistencia": "50%",
        "estado":     "Riesgo académico",
    },
}

NORMATIVAS = {
    "matricula":       "La matrícula se realiza dentro del cronograma oficial. Extemporánea: recargo del 10%.",
    "convalidacion":   "Solicitud con notas mín. 14 del curso original. Plazo: hasta 2da semana del ciclo.",
    "reincorporacion": "Estudiantes con licencia: trámite en Coordinación. Máx. 2 ciclos consecutivos.",
    "titulacion":      "Modalidades: tesis, suficiencia profesional, examen de competencias. 4 créditos mín.",
    "beca":            "Beca Institución Educativa: promedio mínimo 16, sin desaprobados. Renovación semestral.",
}

consulta_count = 0

# ═══════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════

@mcp.tool
def consultar_estudiante(codigo: str) -> str:
    """Consulta los datos académicos de un estudiante de Institución Educativa.

    Usar cuando se pregunte por información de un alumno específico:
    notas, asistencia, carrera, ciclo, situación académica.

    Args:
        codigo: Código del estudiante, formato 'TXXXXX'.
                Códigos disponibles: T20231, T20232, T20233.
    """
    codigo_norm = codigo.strip().upper()
    if codigo_norm not in ESTUDIANTES:
        return (f"Estudiante '{codigo}' NO EXISTE en el sistema. "
                f"Códigos válidos: {', '.join(ESTUDIANTES.keys())}.")

    e = ESTUDIANTES[codigo_norm]
    return (
        f"Estudiante: {codigo_norm} - {e['nombre']}\n"
        f"Carrera: {e['carrera']}\n"
        f"Ciclo: {e['ciclo']}\n"
        f"Promedio: {e['promedio']}\n"
        f"Asistencia: {e['asistencia']}\n"
        f"Estado: {e['estado']}"
    )

@mcp.tool
def buscar_normativa(tema: str) -> str:
    """Busca normativas y reglamentos académicos de Institución Educativa.

    Usar para preguntas sobre procedimientos, requisitos y reglamentos.

    Args:
        tema: Tema a consultar. Temas disponibles: matrícula,
              convalidación, reincorporación, titulación, beca.
    """
    tema_norm = normalizar(tema)
    results = [v for k, v in NORMATIVAS.items() if k in tema_norm]

    if not results:
        return (f"NO HAY normativa sobre '{tema}'. "
                f"Temas disponibles: {', '.join(NORMATIVAS.keys())}.")
    return "\n".join(results)

# ═══════════════════════════════════════════════════════════
# RESOURCES
# ═══════════════════════════════════════════════════════════

@mcp.resource("catalogo://carreras")
def get_carreras() -> str:
    """Retorna el catálogo completo de carreras de Institución Educativa."""
    carreras = {
        "software":   {"nombre": "Desarrollo de Software",        "ciclos": 6, "creditos": 120},
        "redes":      {"nombre": "Redes y Comunicaciones",        "ciclos": 6, "creditos": 118},
        "datos":      {"nombre": "Big Data y Ciencia de Datos",   "ciclos": 6, "creditos": 120},
        "industrial": {"nombre": "Automatización Industrial",     "ciclos": 6, "creditos": 124},
    }
    return json.dumps(carreras, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════
# PROMPTS — templates reutilizables
# ═══════════════════════════════════════════════════════════
@mcp.prompt
def evaluar_riesgo_academico(codigo_estudiante: str) -> str:
    """Template guiado para evaluar el riesgo académico de un estudiante."""
    return (
        f"Evalúa la situación académica del estudiante {codigo_estudiante}.\n"
        f"Pasos:\n"
        f"1) Consulta sus datos con consultar_estudiante.\n"
        f"2) Si el promedio es <13 o la asistencia <80%, identifica el riesgo.\n"
        f"3) Busca normativas relevantes con buscar_normativa (ej: reincorporación)."

    )


# ═══════════════════════════════════════════════════════════
# EJECUTAR EL SERVER
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    #print(consultar_estudiante("T20231"))

    mcp.run(transport="stdio")