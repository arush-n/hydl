# syntax=docker/dockerfile:1.7
#
# One way to start the system, on any machine.
#
#   docker compose up console            # CPU, http://127.0.0.1:8770
#   docker compose --profile gpu up console-gpu
#
# TWO TARGETS, NOT ONE IMAGE WITH A FLAG. `cpu` and `gpu` need different base
# images and different JAX wheels; a single image that tries to be both either
# carries a CUDA runtime nobody asked for or resolves the wrong wheel at start.
#
# THE SOURCE TREE IS NOT PIP-INSTALLED. `console.core.storage` derives
# PROJECT_ROOT from its own file location (`parents[2]`), and `hytalegym` is not
# on any index -- it ships at HytaleRL/hytalegym and goes on PYTHONPATH. An
# installed copy in site-packages would put PROJECT_ROOT somewhere that is not
# the repository and silently split the store from the code. So the image runs
# from /app with PYTHONPATH set, which is exactly what README.md tells a local
# user to do. Container and laptop then behave the same way, which is the whole
# point of normalising the launch.


# --------------------------------------------------------------------------
# Third-party requirements, derived from pyproject rather than restated.
#
# A hand-copied dependency list in a Dockerfile is a second source of truth that
# drifts the first time someone bumps a pin. This stage reads the real one. It
# is also its own layer, so editing source does not reinstall JAX.
# --------------------------------------------------------------------------
FROM python:3.12-slim AS requirements
WORKDIR /src
COPY pyproject.toml ./
RUN <<'PY' python3
import tomllib, pathlib
project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]
wanted = list(project["dependencies"])
wanted += project["optional-dependencies"]["console"]
pathlib.Path("/requirements.txt").write_text("\n".join(wanted) + "\n")
print("\n".join(wanted))
PY


# --------------------------------------------------------------------------
# Shared runtime layout. Both targets end up identical below this line.
# --------------------------------------------------------------------------
FROM python:3.12-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # `hytalegym` is resolved from the tree, not from an index.
    PYTHONPATH=/app:/app/HytaleRL \
    # MUST be 0.0.0.0 in a container. The default is loopback, which inside a
    # container is the container's own loopback -- `-p 8770:8770` would publish
    # a port with nothing listening on it and the server would look healthy in
    # its own logs while every request was refused.
    HYTALERL_CONSOLE_HOST=0.0.0.0 \
    HYTALERL_CONSOLE_PORT=8770 \
    # Every console-written artifact goes here, which is a volume. Baking run
    # output into a layer would make the image a snapshot of one session.
    HYTALERL_STORAGE_ROOT=/data

COPY --from=requirements /requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

# Runs as a non-root user, and /data is chowned to it BEFORE the volume is
# declared -- a volume created from an image inherits the mount point's
# ownership, so doing this after would leave the console unable to write its
# own store.
RUN useradd --create-home --uid 10001 hydl \
 && mkdir -p /data \
 && chown -R hydl:hydl /data /app
USER hydl
VOLUME ["/data"]

EXPOSE 8770

# `requests` is not a dependency, so the check uses the standard library. It
# asks the app, not the port: a bound socket with a broken application is
# exactly the failure a port check cannot see.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/api/version', timeout=4).status==200 else 1)"]

# `--no-reload` because uvicorn's reloader watches the source tree and forks a
# child; under a container's PID 1 that makes signals and shutdown unreliable,
# and there is nothing to reload in an image anyway.
CMD ["python", "-m", "console.server", "--no-reload"]


# --------------------------------------------------------------------------
# GPU target. Same layout, CUDA base, CUDA-enabled JAX.
#
# The base tag is an ARG because it has to match the host's driver, and the
# right answer is a property of the machine rather than of this repository.
# Ubuntu 24.04 is chosen for its system Python 3.12; the 22.04 images ship 3.10,
# which is below this project's `requires-python = ">=3.11"`.
# --------------------------------------------------------------------------
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:/app/HytaleRL \
    HYTALERL_CONSOLE_HOST=0.0.0.0 \
    HYTALERL_CONSOLE_PORT=8770 \
    HYTALERL_STORAGE_ROOT=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python

# A venv rather than --break-system-packages: Ubuntu 24.04 marks its system
# Python externally managed, and overriding that puts pip and apt in charge of
# the same files.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv

COPY --from=requirements /requirements.txt /tmp/requirements.txt
# CUDA JAX first, then the shared requirements. `jax` appears in both; pip keeps
# the already-satisfied CUDA build because the plain `jax>=...` specifier the
# project declares is satisfied by it.
RUN pip install --no-cache-dir "jax[cuda12]>=0.4.35" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

RUN useradd --create-home --uid 10001 hydl \
 && mkdir -p /data \
 && chown -R hydl:hydl /data /app
USER hydl
VOLUME ["/data"]

EXPOSE 8770

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/api/version', timeout=4).status==200 else 1)"]

CMD ["python", "-m", "console.server", "--no-reload"]
