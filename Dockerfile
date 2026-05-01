# Используем легковесный Python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt boto3

COPY src/ src/
COPY run_pipeline.sh .

RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]