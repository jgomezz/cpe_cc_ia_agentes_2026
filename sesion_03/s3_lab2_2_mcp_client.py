"""
Lab 2.2 — Probar el MCP Server desde un cliente (Institución Educativa)
─────────────────────────────────────────────────────────
"""

import asyncio
from fastmcp import Client


async def test_server():

    async with Client("sesion_03/s3_lab2_1_mcp_server.py") as client:

        # ── 1. Descubrir capacidades ──
        tools = await client.list_tools()
        resources = await client.list_resources()
        
        print(f"\n🔧 Tools disponibles ({len(tools)}):")
        for t in tools:
            print(f"   • {t.name}: {t.description[:55]}...")

        print(f"\n📦 Resources disponibles ({len(resources)}):")
        for r in resources:
            print(f"   • {r.uri}")

        # ── 2. Probar consultar_estudiante (caso éxito) ──
        print("\n" + "─" * 60)
        print("TEST: consultar_estudiante('T20231')")
        result = await client.call_tool("consultar_estudiante", {"codigo": "T20231"})
        print(f"  → {result}")

        # ── 3. Leer un Resource estático ──
        print("\n" + "─" * 60)
        print("TEST: resource catalogo://carreras")
        result = await client.read_resource("catalogo://carreras")
        print(f"  → {result}")


if __name__ == "__main__":

    asyncio.run(test_server())