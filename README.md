# IN/OUT X Project Guide

## PostgreSQL Deployment and Data Migration

The application stores its existing document-shaped records in PostgreSQL `JSONB`, with PostgreSQL indexes and transactional writes. Configure `DATABASE_URL` in the host's secret/environment settings using the connection URL from a managed PostgreSQL provider. Remote connections use TLS; do not commit database URLs or credentials. The application creates its table and core indexes at startup, so the database user must be allowed to create tables and indexes.

Keep migration-only MongoDB packages out of the application environment. Create a separate migration environment and configure the source URL locally:

```bash
python3.13 -m venv .migration-venv
.migration-venv/bin/python -m pip install -r requirements-migration.txt
export MONGODB_URI='mongodb+srv://...'
.migration-venv/bin/python mongo_to_postgres.py
```

The default command inventories MongoDB collection names and document counts without writing data. Run `.migration-venv/bin/python mongo_to_postgres.py --apply` to copy all documents in batches, then `... mongo_to_postgres.py --verify` to compare every collection. The migration is rerunnable, preserves MongoDB IDs as strings, and never deletes or modifies source records. For the final cutover, stop both old Mongo-backed services, run `--apply` and `--verify` again, then restart only after both pass. Keep MongoDB available as a rollback/backup until the new services are confirmed healthy.

For a hosted deployment, add `DATABASE_URL` and `POSTGRES_POOL_SIZE` (optional, defaults to `20`) to the web service's secret environment settings and deploy with `requirements-render.txt`. Use a managed PostgreSQL instance in the same region as the web service, enable provider backups, and restrict network access to the application service.

## 1. Python Environment Setup

Run these commands from the repository root. The full requirements include the camera/OCR machine-learning stack and can consume several gigabytes; check free space before installing them.

```bash
cd "/home/iqac2/Documents/INOUTX-FINAL/INOUTX."
df -h .
python3.13 -m venv venv313
source venv313/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
```

The full requirements pin CPU-only PyTorch/Torchvision wheels, avoiding multi-gigabyte CUDA runtime downloads on this host. Camera inference runs on CPU and may be slower than a configured GPU.

For a web-only environment, use:

```bash
python3.13 -m venv venv313
source venv313/bin/activate
python -m pip install -r requirements-render.txt
```

The web systemd service runs Flask-SocketIO under Gunicorn's threaded worker. `SECRET_KEY` is mandatory; startup fails if it is missing. `SESSION_COOKIE_SECURE` defaults to enabled when `APP_BASE_URL` uses HTTPS. Credentialed CORS is restricted to `APP_BASE_URL` by default; set `CORS_ORIGINS` to a comma-separated list if the browser frontend uses additional origins. PyMongo is excluded from normal runtime installs and belongs only to `requirements-migration.txt`.

Do not restart the camera service until the camera/OCR dependencies are installed and validated.

Keep the root `.env` private with `chmod 600 .env`. A unique `SECRET_KEY` is required for production. If it is missing, generate and append one locally with:

```bash
printf '\nSECRET_KEY=%s\n' "$(python3.13 -c 'import secrets; print(secrets.token_urlsafe(48))')" >> .env
```

Activate the environment whenever running commands manually:

```bash
cd "/home/iqac2/Documents/INOUTX-FINAL/INOUTX."
source venv313/bin/activate
```

## 2. Run Manually

Start the web app in one terminal:

```bash
cd "/home/iqac2/Documents/INOUTX-FINAL/INOUTX."
source venv313/bin/activate
cd Frontend
python app.py
```

Start the camera process in a second terminal:

```bash
cd "/home/iqac2/Documents/INOUTX-FINAL/INOUTX."
source venv313/bin/activate
python main.py
```

## 3. Install or Restart Systemd Services

The service files use this checkout's `venv313` path. Install and restart only the web service with:

```bash
cd "/home/iqac2/Documents/INOUTX-FINAL/INOUTX."
sudo ./setup_autostart.sh web
```

After installing all camera/OCR dependencies, install or restart only the camera service:

```bash
sudo ./setup_autostart.sh camera
```

Install and restart both services only when both runtimes are ready:

```bash
sudo ./setup_autostart.sh all
```

For Python code or `.env` changes, restart only the affected service with `sudo systemctl restart inoutx.service` or `sudo systemctl restart inoutx-camera.service`. Run `sudo systemctl daemon-reload` after editing a `.service` file; `setup_autostart.sh` does this automatically.

Check status and logs:

```bash
sudo systemctl status inoutx.service --no-pager
sudo journalctl -u inoutx.service -f
sudo systemctl status inoutx-camera.service --no-pager
sudo journalctl -u inoutx-camera.service -f
```
