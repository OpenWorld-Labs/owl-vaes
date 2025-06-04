FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
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

RUN /bin/bash -c "uv venv && \
    uv pip install ."

ARG DEV_MODE=false
RUN if [ "$DEV_MODE" = "true" ] ; then \
    /bin/bash -c "uv pip install -e '.[dev]'" ; \
    fi

CMD ["/bin/bash", "-c", "source .venv/bin/activate && /bin/bash"]