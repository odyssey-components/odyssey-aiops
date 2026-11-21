FROM python:3.14-slim
WORKDIR /opt/aiops
RUN sed -i 's|http://deb.debian.org|https://mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Languages=none -o Acquire::https::Timeout=30 -o Acquire::Retries=2 update \
    && apt-get install -y --no-install-recommends git libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY site-packages /usr/local/lib/python3.14/site-packages
COPY --chown=10001:10001 go1-source /opt/aiops/go1-source
COPY aiops-worker /usr/local/bin/aiops-worker
ENV HOME=/tmp TMPDIR=/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/aiops-worker"]
