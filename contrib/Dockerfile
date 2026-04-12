FROM python:3.11-slim

WORKDIR /app

# Install build deps for sentence-transformers (numpy/scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e ".[server,cloud]"

# Memory is persisted via a Docker volume mounted at /data/memory
ENV MEMORY_PATH=/data/memory

EXPOSE 8000

CMD ["uvicorn", "memory_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
