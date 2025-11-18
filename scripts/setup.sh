#!/bin/bash
#
# Setup script for CrowdCount Counter Server
# Downloads and installs MediaMTX
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$PROJECT_ROOT/bin"

echo "=================================="
echo "CrowdCount Counter Server Setup"
echo "=================================="
echo ""

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
case $ARCH in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64v8"
        ;;
    armv7l)
        ARCH="armv7"
        ;;
    *)
        echo -e "${RED}Unsupported architecture: $ARCH${NC}"
        exit 1
        ;;
esac

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"
echo ""

# MediaMTX version
MEDIAMTX_VERSION="v1.9.3"
echo "MediaMTX Version: $MEDIAMTX_VERSION"
echo ""

# Construct download URL
MEDIAMTX_FILENAME="mediamtx_${MEDIAMTX_VERSION}_${OS}_${ARCH}.tar.gz"
MEDIAMTX_URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${MEDIAMTX_FILENAME}"

echo "Download URL: $MEDIAMTX_URL"
echo ""

# Create bin directory
mkdir -p "$BIN_DIR"

# Download MediaMTX
echo -e "${YELLOW}Downloading MediaMTX...${NC}"
cd "$BIN_DIR"

if command -v wget &> /dev/null; then
    wget -O mediamtx.tar.gz "$MEDIAMTX_URL"
elif command -v curl &> /dev/null; then
    curl -L -o mediamtx.tar.gz "$MEDIAMTX_URL"
else
    echo -e "${RED}Error: Neither wget nor curl is available${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Download complete${NC}"
echo ""

# Extract
echo -e "${YELLOW}Extracting MediaMTX...${NC}"
tar -xzf mediamtx.tar.gz

# Make executable
chmod +x mediamtx

# Clean up
rm mediamtx.tar.gz

echo -e "${GREEN}✓ MediaMTX installed to $BIN_DIR/mediamtx${NC}"
echo ""

# Check Python dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"

PYTHON_CMD="python3"

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version)"

# Check for required Python packages
REQUIRED_PACKAGES=("cv2:opencv-python" "flask:flask" "requests:requests" "numpy:numpy")
MISSING_PACKAGES=()

for pkg_info in "${REQUIRED_PACKAGES[@]}"; do
    IFS=':' read -r import_name pip_name <<< "$pkg_info"

    if ! $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        MISSING_PACKAGES+=("$pip_name")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing Python packages: ${MISSING_PACKAGES[*]}${NC}"
    echo ""
    echo "Install them with:"
    echo "  pip3 install ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Or install all at once:"
    echo "  pip3 install opencv-python flask requests numpy"
    echo ""
else
    echo -e "${GREEN}✓ All Python dependencies are installed${NC}"
fi

echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Make sure Tailscale is running:"
echo "     tailscale status"
echo ""
echo "  2. Start the server:"
echo "     python3 scripts/run_server.py"
echo ""
echo "  3. On Raspberry Pis, run:"
echo "     python3 stream_to_mediamtx.py <YOUR_PC_TAILSCALE_IP> <stream_name>"
echo ""
echo "Your Tailscale IP:"
if command -v tailscale &> /dev/null; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "Not found")
    echo "  $TAILSCALE_IP"
else
    echo "  (Install Tailscale to see IP)"
fi
echo ""
