FROM python:3.11-slim-bookworm

WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY engine ./engine
COPY bench ./bench
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

CMD ["python", "-c", "print('context-engine image ready')"]
