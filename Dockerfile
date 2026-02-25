FROM python:3.12 AS base

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential ffmpeg libsm6 libxext6

FROM base AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM base AS runner

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . /app

CMD ["bash", "-c", "python3 /app/main.py"]