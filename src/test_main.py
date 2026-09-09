import os
import uuid
from datetime import datetime, timezone
from time import perf_counter
import pandas as pd
from dotenv import load_dotenv

from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI, APIError
import google.genai as genai

# 1. CONFIGURACIÓN DEL ENTORNO CLOUD
PROJECT_ID = "tfm-llm-brand-analysis"
STG_TABLE = f"{PROJECT_ID}.mcdonalds.mcdonalds_queries_stg"
SOURCE_TABLE = f"{PROJECT_ID}.mcdonalds.mcdonalds_queries"
RAW_TABLE = f"{PROJECT_ID}.llm_analysis_bronze.llm_responses_raw"

ITERATIONS_PER_QUERY = 3
TIMEOUT_SECONDS = 60

SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

def init_clients():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(root_dir, '.env'), encoding="utf-8-sig")

    raw_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if raw_cred_path:
        cred_path = os.path.abspath(os.path.join(root_dir, raw_cred_path))
        credentials = service_account.Credentials.from_service_account_file(cred_path, scopes=SCOPES)
        bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    else:
        bq_client = bigquery.Client(project=PROJECT_ID)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=TIMEOUT_SECONDS) if os.getenv("OPENAI_API_KEY") else None
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None

    return bq_client, openai_client, gemini_client

def fetch_openai(client, prompt: str):
    start = perf_counter()
    res = {
        "requested_model": "gpt-4o-mini", "model": "gpt-4o-mini", "response_id": "",
        "response": "", "status": "completed", "incomplete_reason": "",
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "reasoning_effort": "none", "max_output_tokens": 1000,
        "error_type": "", "http_status": None
    }
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        res["response"] = response.choices[0].message.content or ""
        res["response_id"] = response.id or str(uuid.uuid4())
        
        if response.usage:
            res["input_tokens"] = response.usage.prompt_tokens
            res["output_tokens"] = response.usage.completion_tokens
            res["total_tokens"] = response.usage.total_tokens

        if not res["response"].strip():
            res["status"] = "empty_response"

    except APIError as error:
        res["status"] = "error"
        res["error_type"] = type(error).__name__
        res["http_status"] = getattr(error, "status_code", None)
        res["response"] = str(error)
    except Exception as e:
        res["status"] = "error"
        res["error_type"] = type(e).__name__
        res["response"] = str(e)

    res["duration_seconds"] = round(perf_counter() - start, 3)
    return res

def fetch_gemini(client, prompt: str):
    start = perf_counter()
    res = {
        "requested_model": "gemini-2.5-flash", "model": "gemini-2.5-flash", "response_id": str(uuid.uuid4()),
        "response": "", "status": "completed", "incomplete_reason": "",
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "reasoning_effort": "none", "max_output_tokens": 1000,
        "error_type": "", "http_status": None
    }
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        res["response"] = response.text or ""

        if response.usage_metadata:
            res["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            res["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            res["total_tokens"] = res["input_tokens"] + res["output_tokens"]

        if not res["response"].strip():
            res["status"] = "empty_response"

    except Exception as e:
        res["status"] = "error"
        res["error_type"] = type(e).__name__
        res["response"] = str(e)

    res["duration_seconds"] = round(perf_counter() - start, 3)
    return res

def main():
    bq_client, openai_client, gemini_client = init_clients()

    if not openai_client and not gemini_client:
        raise ValueError("No se han configurado API Keys válidas en el entorno.")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Sincronizando tabla nativa con Google Sheets...")
    refresh_query = f"""
        CREATE OR REPLACE TABLE `{SOURCE_TABLE}` AS
        SELECT * FROM `{STG_TABLE}`
    """
    bq_client.query(refresh_query).result()
    print("¡Tabla de queries actualizada con éxito!")

    # AQUI ESTÁ EL CAMBIO: AÑADIMOS "LIMIT 3" A LA CONSULTA SQL
    print(f"[{datetime.now(timezone.utc).isoformat()}] Leyendo las 3 primeras queries desde `{SOURCE_TABLE}` (MODO PRUEBA)...")
    queries_df = bq_client.query(f"SELECT query_id, brand, topic, query_type, query FROM `{SOURCE_TABLE}` LIMIT 3").to_dataframe()

    if queries_df.empty:
        print("La tabla de origen está vacía.")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_TEST"
    started_at_utc = datetime.now(timezone.utc)
    records = []

    print(f"Iniciando ejecucion de prueba run_id: {run_id} | Queries a procesar: {len(queries_df)}")

    for idx, row in queries_df.iterrows():
        print(f"Procesando [{idx+1}/{len(queries_df)}] Query ID: {row['query_id']}")

        for iteration in range(1, ITERATIONS_PER_QUERY + 1):
            
            if openai_client:
                oai = fetch_openai(openai_client, row["query"])
                records.append({
                    "query_id": row["query_id"], "brand": row["brand"], "topic": row["topic"],
                    "query_type": row["query_type"], "query": row["query"], "run_id": run_id,
                    "requested_model": oai["requested_model"], "model": oai["model"], "response_id": oai["response_id"],
                    "response": oai["response"], "status": oai["status"], "incomplete_reason": oai["incomplete_reason"],
                    "input_tokens": oai["input_tokens"], "output_tokens": oai["output_tokens"], "total_tokens": oai["total_tokens"],
                    "started_at_utc": started_at_utc, "duration_seconds": oai["duration_seconds"], "reasoning_effort": oai["reasoning_effort"],
                    "max_output_tokens": oai["max_output_tokens"], "error_type": oai["error_type"], "http_status": oai["http_status"]
                })

            if gemini_client:
                gem = fetch_gemini(gemini_client, row["query"])
                records.append({
                    "query_id": row["query_id"], "brand": row["brand"], "topic": row["topic"],
                    "query_type": row["query_type"], "query": row["query"], "run_id": run_id,
                    "requested_model": gem["requested_model"], "model": gem["model"], "response_id": gem["response_id"],
                    "response": gem["response"], "status": gem["status"], "incomplete_reason": gem["incomplete_reason"],
                    "input_tokens": gem["input_tokens"], "output_tokens": gem["output_tokens"], "total_tokens": gem["total_tokens"],
                    "started_at_utc": started_at_utc, "duration_seconds": gem["duration_seconds"], "reasoning_effort": gem["reasoning_effort"],
                    "max_output_tokens": gem["max_output_tokens"], "error_type": gem["error_type"], "http_status": gem["http_status"]
                })

    if records:
        output_df = pd.DataFrame(records)
        print(f"Volcando {len(output_df)} filas en `{RAW_TABLE}`...")
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = bq_client.load_table_from_dataframe(output_df, RAW_TABLE, job_config=job_config)
        job.result()
        print("¡Proceso de PRUEBA completado y guardado en BigQuery con éxito!")

if __name__ == "__main__":
    main()