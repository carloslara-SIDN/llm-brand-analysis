import json
import os
import re
import unicodedata
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI, APIError
from pydantic import BaseModel, ConfigDict, ValidationError

# 1. CONFIGURACIÓN DEL ENTORNO CLOUD
PROJECT_ID = "tfm-llm-brand-analysis"
BRONZE_TABLE = f"{PROJECT_ID}.llm_analysis_bronze.llm_responses_raw"
SILVER_TABLE = f"{PROJECT_ID}.llm_analysis_silver.mcdonalds_responses_analyzed"

ANALYSIS_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "brand_analysis_v5"
ALIAS_VERSION = "brand_aliases_v3"
MAX_VALIDATION_RETRIES = 1

SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# 2. DICCIONARIOS DE MARCAS Y NORMALIZACIÓN
BRAND_ALIASES = {
    "McDonald's": ["McDonald’s", "McDonalds", "Mc Donald’s", "Mc Donalds", "McDonald", "MacDonalds", "McDonals"],
    "Burger King": ["BurgerKing", "BK", "B.K.", "Burguer King", "Burgeur King", "Burgeuer King"],
    "Five Guys": ["FiveGuys", "5 Guys", "Five guys", "five guys"],
    "Carl’s Jr.": ["Carl’s Jr", "Carls Jr", "CarlsJr", "Carl’s Junior", "Carls Junior"],
    "Goiko": ["Goiko Grill", "GoikoGrill"],
    "Taco Bell": ["TacoBell", "Taco Bell", "taco bell"],
    "Popeyes": ["Popeye’s", "Popeyes Louisiana Kitchen"],
    "KFC": ["K.F.C.", "Kentucky Fried Chicken"],
    "Domino’s Pizza": ["Dominos Pizza", "Domino’s", "Dominos", "Domino Pizza"],
    "Telepizza": ["Tele Pizza", "Telepizza"],
    "Pizza Hut": ["PizzaHut"],
    "Subway": ["Sub Way", "subway", "Subway."],
    "TGB": ["The Good Burger", "TGB (The Good Burger)", "TGB — The Good Burger", "TGB - The Good Burger", "TGB – The Good Burger", "TGB The Good Burger"],
    "Papa John’s": ["Papa Johns", "Papa John's", "PapaJohns", "Papa John’s Pizza"],
    "100 Montaditos": ["100Montaditos", "Cien Montaditos"],
}

EXCLUDED_ENTITIES = [
    "Glovo", "Uber Eats", "UberEats", "Just Eat", "JustEat", "DoorDash", "Door Dash", "Deliveroo",
    "Mercadona", "Lidl", "Carrefour", "Aldi", "Cadenas locales", "Cadenas regionales", "Restaurantes locales",
    "Cadenas de comida rápida", "Restaurantes de comida rápida", "Hamburgueserías", "Pizzerías", "Supermercados"
]

def normalize_key(value):
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"['’‘ʼ`.]", "", value)
    return " ".join(value.split())

ALIAS_LOOKUP = {}
for canonical, aliases in BRAND_ALIASES.items():
    for alias in [canonical, *aliases]:
        ALIAS_LOOKUP[normalize_key(alias)] = canonical

EXCLUDED_KEYS = {normalize_key(name) for name in EXCLUDED_ENTITIES}

def normalize_brand(name):
    return ALIAS_LOOKUP.get(normalize_key(name), str(name).strip())

def normalize_search_text(value):
    return normalize_key(re.sub(r"[*_~]", "", str(value)))

def first_brand_position(name, normalized_response):
    positions = []
    for variant in [name, *BRAND_ALIASES.get(name, [])]:
        key = normalize_search_text(variant)
        if not key:
            continue
        match = re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", normalized_response)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None

def known_brand_positions(response_text):
    normalized_response = normalize_search_text(response_text)
    found = {}
    for name in BRAND_ALIASES:
        position = first_brand_position(name, normalized_response)
        if position is not None:
            found[name] = position
    return found

def clean_brands(names, target_brand, response_text):
    cleaned = []
    seen = set()
    target = normalize_brand(target_brand)

    for name in names:
        if not str(name).strip(): continue
        if normalize_key(name) in EXCLUDED_KEYS: continue
        canonical = normalize_brand(name)
        if normalize_key(canonical) == normalize_key(target): canonical = target
        key = normalize_key(canonical)
        if key not in seen:
            cleaned.append(canonical)
            seen.add(key)

    normalized_response = normalize_search_text(response_text)
    positions = known_brand_positions(response_text)
    for name in cleaned:
        position = first_brand_position(name, normalized_response)
        if position is not None:
            positions[name] = position
            
    return sorted(cleaned, key=lambda name: (positions.get(name, 999999), -len(name)))

# 3. SCHEMA DE PYDANTIC
class BrandAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brand_mentioned: bool
    brands_in_order: list[str]
    positive_evidence_ids: list[str]
    negative_evidence_ids: list[str]
    sentiment_reason: str
    recommendation_type: str | None
    recommendation_evidence_ids: list[str]
    recommendation_reason: str
    attributes: list[str]
    evidence_ids: list[str]

EVIDENCE_ID_FIELDS = ("evidence_ids", "positive_evidence_ids", "negative_evidence_ids", "recommendation_evidence_ids")

def make_fragments(response_text):
    fragments = {}
    offset = 0
    for line in str(response_text).splitlines(keepends=True):
        text = line.rstrip("\r\n")
        if text.strip():
            fragment_id = f"F{len(fragments) + 1:03d}"
            fragments[fragment_id] = {"text": text, "start": offset, "end": offset + len(text)}
        offset += len(line)
    return fragments

def derive_sentiment(analysis):
    if not analysis.brand_mentioned: return None
    positive = bool(analysis.positive_evidence_ids)
    negative = bool(analysis.negative_evidence_ids)
    if positive and negative: return "mixto"
    if positive: return "positivo"
    if negative: return "negativo"
    return "neutro"

# 4. CLIENTES
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

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)
    return bq_client, openai_client

# 5. CONSULTA INCREMENTAL
def fetch_unprocessed_responses(bq_client):
    query = f"""
        SELECT b.response_id, b.query_id, b.brand, b.query, b.response, b.llm_model
        FROM `{BRONZE_TABLE}` b
        LEFT JOIN `{SILVER_TABLE}` s ON b.response_id = s.response_id
        WHERE s.response_id IS NULL AND b.status = 'completed' AND TRIM(b.response) != ''
    """
    return bq_client.query(query).to_dataframe()

# 6. ANÁLISIS INDIVIDUAL CON RETRIES Y TELEMETRÍA
def analyze_single_response(openai_client, row, target_brand, run_id):
    fragments = make_fragments(row["response"])
    if not fragments: return None

    response_schema = BrandAnalysis.model_json_schema()
    for field in EVIDENCE_ID_FIELDS:
        response_schema["properties"][field]["items"] = {"type": "string", "enum": list(fragments)}

    detected = known_brand_positions(row["response"])
    payload_base = {
        "target_brand": target_brand,
        "question": row["query"],
        "detected_known_brands": sorted(detected, key=detected.get),
        "answer_fragments": [{"id": f_id, "text": f["text"]} for f_id, f in fragments.items()]
    }

    record = {
        "query_id": row["query_id"], "response_id": row["response_id"], "brand": target_brand,
        "query": row["query"], "response": row["response"], "llm_model": row["llm_model"],
        "analysis_run_id": run_id, "source_file": "cloud_run_incremental",
        "prompt_version": PROMPT_VERSION, "alias_version": ALIAS_VERSION,
        "analysis_model": ANALYSIS_MODEL, "analysis_input_tokens": 0,
        "analysis_output_tokens": 0, "analysis_total_tokens": 0,
        "analysis_attempts": 0, "attempt_history": [], "error_type": "", "error_message": "", "http_status": None
    }

    history = []
    repair_context = None
    success = False

    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
        record["analysis_attempts"] = attempt
        attempt_info = {"attempt": attempt, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        
        payload = dict(payload_base)
        if repair_context: payload["repair_context"] = repair_context

        try:
            response = openai_client.chat.completions.create(
                model=ANALYSIS_MODEL,
                messages=[
                    {"role": "system", "content": "Analiza cómo aparece la marca objetivo en la respuesta según las reglas de extracción estricta."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "brand_analysis", "strict": True, "schema": response_schema}},
                temperature=0.0
            )
            
            record["analysis_response_id"] = response.id
            record["raw_analysis"] = response.choices[0].message.content or ""
            
            if response.usage:
                record["analysis_input_tokens"] += response.usage.prompt_tokens
                record["analysis_output_tokens"] += response.usage.completion_tokens
                record["analysis_total_tokens"] += response.usage.total_tokens

            analysis = BrandAnalysis.model_validate_json(record["raw_analysis"])
            analysis.brands_in_order = clean_brands(analysis.brands_in_order, target_brand, row["response"])
            
            combined_ids = []
            for field in EVIDENCE_ID_FIELDS:
                combined_ids.extend(getattr(analysis, field))
            combined_ids = list(dict.fromkeys(combined_ids))
            evidence_texts = [fragments[f_id]["text"] for f_id in combined_ids if f_id in fragments]

            record.update({
                "brand_mentioned": analysis.brand_mentioned,
                "brands_in_order": json.dumps(analysis.brands_in_order, ensure_ascii=False),
                "mention_position": (analysis.brands_in_order.index(target_brand) + 1) if (analysis.brand_mentioned and target_brand in analysis.brands_in_order) else None,
                "sentiment": derive_sentiment(analysis),
                "recommended": (analysis.recommendation_type == "opcion_recomendada") if analysis.brand_mentioned else None,
                "attributes": json.dumps(analysis.attributes, ensure_ascii=False),
                "evidence": json.dumps(evidence_texts, ensure_ascii=False),
                "evidence_ids": json.dumps(combined_ids, ensure_ascii=False),
                "positive_evidence_ids": json.dumps(analysis.positive_evidence_ids, ensure_ascii=False),
                "negative_evidence_ids": json.dumps(analysis.negative_evidence_ids, ensure_ascii=False),
                "sentiment_reason": analysis.sentiment_reason,
                "recommendation_type": analysis.recommendation_type,
                "recommendation_evidence_ids": json.dumps(analysis.recommendation_evidence_ids, ensure_ascii=False),
                "recommendation_reason": analysis.recommendation_reason,
                "analysis_status": "validated",
                "analyzed_at_utc": datetime.now(timezone.utc).isoformat()
            })
            success = True
            attempt_info.update({"status": "validated"})
            history.append(attempt_info)
            break

        except APIError as error:
            code = getattr(error, "status_code", None)
            record["analysis_status"] = "api_error"
            record["error_type"] = type(error).__name__
            record["http_status"] = code
            record["error_message"] = str(error)
            attempt_info.update({"status": "api_error", "error": record["error_message"]})
            history.append(attempt_info)
            if code in {400, 401, 403, 404, 429}: break

        except (ValidationError, ValueError) as error:
            record["analysis_status"] = "validation_error"
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
            attempt_info.update({"status": "validation_error", "error": record["error_message"]})
            history.append(attempt_info)
            repair_context = {"previous_analysis": record.get("raw_analysis", ""), "validation_error": record["error_message"]}

    record["attempt_history"] = json.dumps(history, ensure_ascii=False)
    
    # Si fallaron todos los intentos, guardamos los datos vacíos correspondientes
    if not success:
        record["analyzed_at_utc"] = datetime.now(timezone.utc).isoformat()
        
    return record

# 7. EJECUCIÓN PRINCIPAL
def main():
    bq_client, openai_client = init_clients()
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Buscando respuestas pendientes de análisis en Bronze...")
    unprocessed_df = fetch_unprocessed_responses(bq_client)

    if unprocessed_df.empty:
        print("No hay respuestas pendientes por analizar. La capa Silver está al día.")
        return

    print(f"Respuestas pendientes encontradas: {len(unprocessed_df)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    analyzed_records = []

    for idx, row in unprocessed_df.iterrows():
        target_brand = normalize_brand(row["brand"])
        print(f"Analizando [{idx+1}/{len(unprocessed_df)}] Response ID: {row['response_id']}...")
        
        parsed_record = analyze_single_response(openai_client, row, target_brand, run_id)
        if parsed_record:
            analyzed_records.append(parsed_record)

    if analyzed_records:
        silver_df = pd.DataFrame(analyzed_records)
        print(f"Insertando {len(silver_df)} registros validados en `{SILVER_TABLE}`...")
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        job = bq_client.load_table_from_dataframe(silver_df, SILVER_TABLE, job_config=job_config)
        job.result()
        print("¡Análisis estructurado y telemetría completados y guardados en BigQuery!")

if __name__ == "__main__":
    main()