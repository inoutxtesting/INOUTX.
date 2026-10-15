#!/bin/bash
# Script to set up IN/OUT X app (Frontend & Camera) to run automatically on system boot.
set -euo pipefail

WEB_SERVICE="inoutx.service"
CAMERA_SERVICE="inoutx-camera.service"
TARGET="${1:-all}"

case "$TARGET" in
    web)
        SERVICES=("$WEB_SERVICE")
        ;;
    camera)
        SERVICES=("$CAMERA_SERVICE")
        ;;
    all)
        SERVICES=("$WEB_SERVICE" "$CAMERA_SERVICE")
        ;;
    *)
        echo "Usage: $0 [web|camera|all]"
        exit 2
        ;;
esac

DIR="$(dirname "$(realpath "$0")")"

echo "=================================================="
echo "Installing IN/OUT X Systemd Services for Auto-start"
echo "=================================================="

for SERVICE in "${SERVICES[@]}"; do
    echo "Installing $SERVICE..."
    sudo cp "$DIR/$SERVICE" "/etc/systemd/system/$SERVICE"
    sudo chmod 644 "/etc/systemd/system/$SERVICE"
done

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

sudo systemctl enable "${SERVICES[@]}"
sudo systemctl restart "${SERVICES[@]}"

echo ""
echo "Checking service statuses..."
for SERVICE in "${SERVICES[@]}"; do
    echo "--- $SERVICE ---"
    sudo systemctl status "$SERVICE" --no-pager
done

echo "=================================================="
echo "Setup complete! Selected service(s) are enabled and restarted."
echo "Useful commands:"
echo "  Check Web App status   : sudo systemctl status inoutx.service"
echo "  Check Camera status    : sudo systemctl status inoutx-camera.service"
echo "  View Camera AI logs    : sudo journalctl -u inoutx-camera.service -f"
echo "  View Web App logs      : sudo journalctl -u inoutx.service -f"
echo "  Configure/restart one  : sudo ./setup_autostart.sh [web|camera|all]"
echo "=================================================="
