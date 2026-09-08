import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "tfm-llm-brand-analysis"
STG_TABLE = f"{PROJECT_ID}.mcdonalds.mcdonalds_queries_stg"
NATIVE_TABLE = f"{PROJECT_ID}.mcdonalds.mcdonalds_queries"

# Ámbitos requeridos para consultar tablas externas conectadas a Google Sheets / Drive
SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

def test_bigquery_connection():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, '.env')
    
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")

    raw_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not raw_cred_path:
        raise ValueError("No se encontró la variable GOOGLE_APPLICATION_CREDENTIALS en el archivo .env")

    cred_path = os.path.abspath(os.path.join(root_dir, raw_cred_path))
    
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"El archivo JSON no existe en la ruta: {cred_path}")

    print(f"Cargando credenciales con scopes de Drive/BigQuery desde: {cred_path}")
    
    # Añadimos scopes explícitamente para permitir la lectura de Google Sheets
    credentials = service_account.Credentials.from_service_account_file(
        cred_path,
        scopes=SCOPES
    )
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    print("Refrescando datos desde Google Sheets a la tabla nativa...")
    refresh_query = f"""
        CREATE OR REPLACE TABLE `{NATIVE_TABLE}` AS
        SELECT * FROM `{STG_TABLE}`
    """
    bq_client.query(refresh_query).result()
    print("¡Tabla de BigQuery sincronizada correctamente desde Google Sheets!")

    read_query = f"SELECT * FROM `{NATIVE_TABLE}`"
    df = bq_client.query(read_query).to_dataframe()

    print("\n" + "="*40)
    print(" DIAGNÓSTICO DE CONEXIÓN Y DATOS")
    print("="*40)
    print(f"Número total de filas: {len(df)}")
    print(f"Número total de columnas: {len(df.columns)}")
    print("\nColumnas detectadas:")
    for col in df.columns:
        print(f" - {col}")
    
    print("\nVista previa de las primeras 3 filas:")
    print(df.head(3))
    print("="*40)

if __name__ == "__main__":
    test_bigquery_connection()