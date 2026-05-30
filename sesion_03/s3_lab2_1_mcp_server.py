"""
Lab 2.1 — MCP Server con FastMCP (Institución Educativa)
──────────────────────────────────────────

import json

"""

import unicodedata
from fastmcp import FastMCP

# Crear el servidor MCP
mcp = FastMCP("Institución Educativa Asistente Académico")




# ═══════════════════════════════════════════════════════════
# EJECUTAR EL SERVER
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")