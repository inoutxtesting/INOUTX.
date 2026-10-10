#!/bin/bash
# Script to set up IN/OUT X app (Frontend & Camera) to run automatically on system boot.

WEB_SERVICE="inoutx.service"
CAMERA_SERVICE="inoutx-camera.service"

DIR="$(dirname "$(realpath "$0")")"

echo "=================================================="
echo "Installing IN/OUT X Systemd Services for Auto-start"
echo "=================================================="

# Copy web service
if [ -f "$DIR/$WEB_SERVICE" ]; then
    echo "Installing $WEB_SERVICE..."
    sudo cp "$DIR/$WEB_SERVICE" "/etc/systemd/system/$WEB_SERVICE"
    sudo chmod 644 "/etc/systemd/system/$WEB_SERVICE"
    sudo systemctl enable "$WEB_SERVICE"
    sudo systemctl restart "$WEB_SERVICE"
fi

# Copy camera service
if [ -f "$DIR/$CAMERA_SERVICE" ]; then
    echo "Installing $CAMERA_SERVICE..."
    sudo cp "$DIR/$CAMERA_SERVICE" "/etc/systemd/system/$CAMERA_SERVICE"
    sudo chmod 644 "/etc/systemd/system/$CAMERA_SERVICE"
    sudo systemctl enable "$CAMERA_SERVICE"
    sudo systemctl restart "$CAMERA_SERVICE"
fi

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "Checking service statuses..."
echo "--- Web App Service ---"
sudo systemctl status "$WEB_SERVICE" --no-pager
echo ""
echo "--- Camera AI Service ---"
sudo systemctl status "$CAMERA_SERVICE" --no-pager

echo "=================================================="
echo "Setup complete! Both Web App and Camera AI service will start on boot."
echo "Useful commands:"
echo "  Check Web App status   : sudo systemctl status inoutx.service"
echo "  Check Camera status    : sudo systemctl status inoutx-camera.service"
echo "  View Camera AI logs    : sudo journalctl -u inoutx-camera.service -f"
echo "  View Web App logs      : sudo journalctl -u inoutx.service -f"
echo "=================================================="
