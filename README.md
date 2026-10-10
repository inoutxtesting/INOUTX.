# IN/OUT X Project Guide

## 1. Virtual Environment Activation & Manual Running

To activate the Python virtual environment:
```bash
source /home/iqac2/Documents/INOUTX/venv313/bin/activate
```

To run the Camera Detection manually:
```bash
python3 camera.py
```

To run the Web Frontend App manually:
```bash
cd Frontend
python3 app.py
```

---

## 2. Auto-start on System Boot (Systemd Services)

Both the Web App and Camera AI Detection services are configured to run automatically on machine boot.

To install or update autostart services:
```bash
sudo ./setup_autostart.sh
```

### Management Commands:
- **Camera AI Service**:
  - Check status: `sudo systemctl status inoutx-camera.service`
  - View live logs: `sudo journalctl -u inoutx-camera.service -f`
  - Restart: `sudo systemctl restart inoutx-camera.service`
  - Stop: `sudo systemctl stop inoutx-camera.service`

- **Web App Service**:
  - Check status: `sudo systemctl status inoutx.service`
  - View live logs: `sudo journalctl -u inoutx.service -f`
  - Restart: `sudo systemctl restart inoutx.service`
  - Stop: `sudo systemctl stop inoutx.service`
