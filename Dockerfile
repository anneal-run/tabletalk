FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY tabletalk ./tabletalk
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt && pip install --no-cache-dir -e .
COPY scripts ./scripts
COPY evals ./evals

ENV TABLETALK_BASE_URL=http://llama:8080/v1
ENTRYPOINT ["tabletalk"]
CMD ["--help"]
