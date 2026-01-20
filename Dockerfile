# CrowdCount Counter Server Docker Image
#
# This image provides the runtime environment for the counter server.
# The actual application code is mounted at runtime via docker-compose volumes,
# allowing different code (prod vs dev) to run in separate containers.

FROM python:3.11-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Video processing
    ffmpeg \
    # Networking tools
    curl \
    wget \
    # Build tools (for some Python packages)
    gcc \
    g++ \
    # envsubst for template substitution
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Download and install MediaMTX
ARG MEDIAMTX_VERSION=1.9.1
RUN mkdir -p /opt/mediamtx && \
    wget -q "https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz" -O /tmp/mediamtx.tar.gz && \
    tar -xzf /tmp/mediamtx.tar.gz -C /opt/mediamtx && \
    rm /tmp/mediamtx.tar.gz && \
    chmod +x /opt/mediamtx/mediamtx

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default environment variables (can be overridden in docker-compose)
ENV RTSP_PORT=8554
ENV ORCHESTRATOR_PORT=5000
ENV HLS_PORT=8888
ENV WEBRTC_PORT=8889
ENV MEDIAMTX_API_PORT=9997
ENV MEDIAMTX_METRICS_PORT=9998
ENV RTSPS_PORT=8555
ENV RTP_PORT=8000
ENV RTCP_PORT=8001
ENV RTMP_PORT=1935

# Expose all ports (actual mapping done in docker-compose)
EXPOSE 8554 8555 5000 8888 8889 9997 9998 1935

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python3", "scripts/run_server.py", "--processor", "counter"]
