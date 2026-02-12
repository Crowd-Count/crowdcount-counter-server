#!/bin/bash
#
# Docker Entrypoint for CrowdCount Counter Server
#
# This script:
# 1. Generates mediamtx.yml from the template with correct port values
# 2. Sets up the MediaMTX binary symlink
# 3. Starts the server
#

set -e

echo "============================================================"
echo "CrowdCount Counter Server - Docker Container"
echo "Environment: ${ENV:-unknown}"
echo "============================================================"

# Create required directories
mkdir -p /app/data/streams
mkdir -p /app/data/recordings
mkdir -p /app/bin

# Set up MediaMTX binary symlink if it doesn't exist
if [ ! -f /app/bin/mediamtx ]; then
    ln -sf /opt/mediamtx/mediamtx /app/bin/mediamtx
    echo "Created MediaMTX symlink"
fi

# Generate mediamtx.yml from template
TEMPLATE_FILE="/app/src/config/mediamtx.yml.template"
OUTPUT_FILE="/app/src/config/mediamtx.yml"

if [ -f "$TEMPLATE_FILE" ]; then
    echo "Generating MediaMTX configuration..."
    echo "  RTSP_PORT: ${RTSP_PORT}"
    echo "  ORCHESTRATOR_PORT: ${ORCHESTRATOR_PORT}"
    echo "  HLS_PORT: ${HLS_PORT}"
    echo "  WEBRTC_PORT: ${WEBRTC_PORT}"
    echo "  MEDIAMTX_API_PORT: ${MEDIAMTX_API_PORT}"
    echo "  MEDIAMTX_METRICS_PORT: ${MEDIAMTX_METRICS_PORT}"

    # Only substitute Docker env vars, preserve $MTX_PATH for MediaMTX runtime
    envsubst '${RTSP_PORT} ${RTSPS_PORT} ${RTP_PORT} ${RTCP_PORT} ${RTMP_PORT} ${HLS_PORT} ${WEBRTC_PORT} ${ORCHESTRATOR_PORT} ${MEDIAMTX_API_PORT} ${MEDIAMTX_METRICS_PORT}' < "$TEMPLATE_FILE" > "$OUTPUT_FILE"
    echo "Generated: $OUTPUT_FILE"
else
    echo "Warning: Template file not found: $TEMPLATE_FILE"
    echo "Using existing mediamtx.yml if available"
fi

echo "============================================================"
echo "Starting server..."
echo "============================================================"

# Execute the command passed to the container
exec "$@"
