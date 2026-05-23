from langchain_core.tools import tool
import requests

from model import get_llm

@tool
def get_exchange_rate(query:str) -> str:
    """
        Busca información actualizada del tipo de cambio del dolar en el BCRP y
        lo devuelve en soles por dolar
    """
    fecha_base = "2026-05-21"
    url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04640PD/json/" + fecha_base 
    data = requests.get(url).json()
    return data["periods"][-1]["values"][0]

tools = [get_exchange_rate]

llm = get_llm(name="ollama")


# LangGraph


if __name__ == "__main__":

    query = "¿Cuál es el tipo de cambio del dolar hoy?"
    response = get_exchange_rate(query)
    print(response)

'''
{ "config": { "title":"Tipo de cambio", 
              "series": [ { "name":"Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta", "dec":"3" }] }, 
              "periods": [ { "name":"21.May.26", "values":["3.418"] } ] }
'''