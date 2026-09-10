FROM python:3.11-slim

WORKDIR /app

# Copiar lista de librerías e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY src/ ./src/

# Ejecutar secuencialmente main.py y analyze_responses.py
CMD ["sh", "-c", "python src/main.py && python src/analyze_responses.py"]