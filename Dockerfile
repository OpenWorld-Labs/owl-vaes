FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    procps \
    git \
    curl \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-venv \
    gcc \
    g++ \
    make \
    autoconf \
    libtool \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN uv venv
RUN uv sync

CMD ["/bin/bash", "-c", "source .venv/bin/activate && /bin/bash"]