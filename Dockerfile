# syntax=docker/dockerfile:1.7
#
# Multi-backend BoltzGen image: CPU, NVIDIA CUDA, and Intel XPU all build
# from this one Dockerfile, mirroring the `cpu`/`cuda`/`xpu` extras in
# pyproject.toml (see the `uv sync --extra ...` calls below).
#
#   docker build --build-arg BACKEND=cpu  -t boltzgen:cpu  .
#   docker build --build-arg BACKEND=cuda -t boltzgen:cuda .
#   docker build --build-arg BACKEND=xpu  -t boltzgen:xpu  .
#
# Run:
#   CPU:  docker run --rm boltzgen:cpu ...
#   CUDA: docker run --rm --gpus all boltzgen:cuda ...
#         (if your nvidia-container-toolkit install doesn't support the CDI
#         path used by --gpus, fall back to
#         `docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all boltzgen:cuda ...`)
#   XPU:  docker run --rm --device /dev/dri \
#           --group-add "$(stat -c '%g' /dev/dri/renderD128)" boltzgen:xpu ...
#         (the render-group GID varies per host, hence `stat`-ing it at run
#         time rather than hardcoding it in the image)

# Declared before the first FROM so it can be used in the final stage's
# `FROM ${BACKEND}-base AS runtime` line below - Docker only allows ARGs
# declared before the *first* FROM to be referenced in later FROM
# instructions, regardless of where those FROM instructions appear.
ARG BACKEND=cuda
ARG UV_VERSION=0.12.9
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

##############################################################################
# base: OS packages + uv, shared by every backend. Deliberately NOT based on
# `nvidia/cuda:...` - torch's CUDA/XPU wheels bundle their own CUDA
# runtime/cuDNN/Level-Zero-adjacent libraries, so a plain Ubuntu base plus a
# handful of small userspace packages (added per-backend below) is enough.
# This also fixes a long-standing cu121-vs-cu130 mismatch between this file
# and pyproject.toml, since CUDA now installs via `uv sync --extra cuda`
# instead of a hardcoded pip index URL.
##############################################################################
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_HTTP_TIMEOUT=300 \
    HF_HOME=/cache

COPY --from=uv /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libgl1 \
    libhdf5-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf "$(command -v python3)" /usr/bin/python

WORKDIR /app

##############################################################################
# xpu-driver: Intel GPU compute-runtime userspace packages (Level-Zero +
# OpenCL ICD only - no video/media packages, BoltzGen doesn't need them).
# Uses the same signed kobuk-team/intel-graphics PPA validated end-to-end on
# real Intel Arc Pro B60 hardware. The i915/xe kernel driver itself is NOT
# needed here - it lives on the host and is reached through /dev/dri.
##############################################################################
FROM base AS xpu-driver
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    libze1 \
    intel-opencl-icd \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

##############################################################################
# Per-backend dependency installation, split into two layers so editing
# application source code doesn't invalidate the slow "download torch"
# layer: step 1 only sees pyproject.toml/uv.lock (via bind mounts, so they
# aren't even persisted in this layer); step 2 copies the real source and
# installs the local project on top of the already-cached dependencies.
##############################################################################
FROM base AS cpu-base
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --extra cpu
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra cpu

FROM base AS cuda-base
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --extra cuda
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra cuda

FROM xpu-driver AS xpu-base
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --extra xpu
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra xpu

##############################################################################
# runtime: shared final stage (weights download + non-root user), built on
# top of whichever backend stage `--build-arg BACKEND=...` selects.
##############################################################################
FROM ${BACKEND}-base AS runtime

ENV PATH="/app/.venv/bin:${PATH}"

ARG DOWNLOAD_WEIGHTS=false
RUN mkdir -p "${HF_HOME}" && \
    if [ "${DOWNLOAD_WEIGHTS}" = "true" ]; then \
        boltzgen download all --cache "${HF_HOME}" --force_download; \
    fi

ARG USERNAME=boltzgen
ARG USER_UID=1000
ARG USER_GID=1000

# `video`/`render` group membership is only exercised by the XPU (and, for
# `render`, some CUDA) device-passthrough path, but adding it unconditionally
# keeps this final stage identical across all three backends. Prefer
# `--group-add "$(stat -c '%g' /dev/dri/renderD128)"` at `docker run` time
# (see header) since the host's actual render-group GID varies by machine.
# Ubuntu's base image ships a preexisting `ubuntu` user/group at UID/GID
# 1000, which collides with our default UID/GID - drop it first so
# `groupadd`/`useradd` below can claim 1000 (or whatever UID/GID is passed).
RUN if getent passwd ${USER_UID} >/dev/null; then userdel -r "$(getent passwd ${USER_UID} | cut -d: -f1)" 2>/dev/null || true; fi && \
    if getent group ${USER_GID} >/dev/null; then groupdel "$(getent group ${USER_GID} | cut -d: -f1)"; fi && \
    groupadd --gid ${USER_GID} ${USERNAME} && \
    groupadd -f render && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} && \
    usermod -aG video,render ${USERNAME} && \
    mkdir -p "${HF_HOME}" && chown -R ${USER_UID}:${USER_GID} "${HF_HOME}"

USER ${USERNAME}
WORKDIR /workspace