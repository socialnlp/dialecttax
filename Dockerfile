# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#######
# UV  #
#######

# uv supplies the Python 3.14 interpreter itself, so no conda/miniconda bootstrap.
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

# copy: the build cache and the venv are on different layers, so hardlinks fail.
# UV_PROJECT_ENVIRONMENT keeps the venv outside the bind-mounted working tree.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace/dialecttax

################
# DEPENDENCIES #
################

# Dependencies resolve in their own layer so editing source does not re-download
# torch. --locked fails loudly if uv.lock is stale rather than silently drifting.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra dev

###########
# PROJECT #
###########

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev

# Put the venv first so `python` is the project interpreter without activation.
ENV PATH="/opt/venv/bin:$PATH"
