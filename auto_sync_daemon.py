#!/usr/bin/env python3
"""
IN/OUT X - Git Auto-Sync Poller Daemon (Vercel-style Auto-Deployment)
=====================================================================
Continuously checks the remote GitHub repository (origin/main) every 30 seconds.
When a new commit is detected:
  1. Pulls the latest code (git pull / reset --hard).
  2. Automatically reloads affected services:
     - Frontend HTML/CSS/JS: live immediately on domain!
     - Backend (app.py): restarts Flask process.
     - Camera Engine (camera.py / main.py): restarts dual camera engine.
     - Dependencies: runs pip install if requirements.txt modified.
  3. Logs all deployment activities and updates deploy_status.json.
"""

import os
import sys
import time
import signal
import subprocess
import json
import logging
from datetime import datetime

# Configuration
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SECONDARY_DIR = "/home/iqac2/Documents/INOUTX"
VENV_PYTHON = "/home/iqac2/Documents/INOUTX/venv313/bin/python3"
VENV_PIP = "/home/iqac2/Documents/INOUTX/venv313/bin/pip"
POLL_INTERVAL_SECONDS = 30
LOG_FILE = os.path.join(os.path.dirname(REPO_DIR), "auto_deploy.log")
STATUS_FILE = os.path.join(REPO_DIR, "deploy_status.json")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("AutoDeploy")

running = True


def signal_handler(signum, frame):
    global running
    logger.info("Received termination signal (%s). Shutting down auto-deploy daemon...", signum)
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_git(args, cwd=REPO_DIR):
    """Executes a git command and returns (success, stdout, stderr)."""
    try:
        res = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=45
        )
        return (res.returncode == 0, res.stdout.strip(), res.stderr.strip())
    except subprocess.TimeoutExpired:
        return (False, "", "Git command timed out")
    except Exception as e:
        return (False, "", str(e))


def restart_process_by_pattern(pattern, start_cmd, env=None):
    """Finds existing process matching pattern, terminates it, and restarts it."""
    try:
        # Find existing PIDs
        pgrep = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        my_pid = str(os.getpid())
        pids = [p for p in pgrep.stdout.strip().split("\n") if p and p != my_pid]

        for pid in pids:
            try:
                logger.info("Stopping old process (%s, PID: %s)...", pattern, pid)
                os.kill(int(pid), signal.SIGTERM)
            except Exception as e:
                logger.warning("Could not terminate PID %s: %s", pid, e)

        # Wait a moment for graceful exit
        time.sleep(2)

        # Force kill if still lingering
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass

        # Spawn new process
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        logger.info("Starting new process: %s", " ".join(start_cmd))
        subprocess.Popen(
            start_cmd,
            cwd=REPO_DIR,
            env=run_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        logger.info("Process '%s' restarted successfully.", pattern)
        return True
    except Exception as e:
        logger.error("Failed to restart '%s': %s", pattern, e)
        return False


def restart_systemd_service(service_name):
    """Restart a systemd-managed service and report failures to the deploy log."""
    try:
        result = subprocess.run(
            ["systemctl", "restart", service_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        logger.info("Restarted systemd service '%s'.", service_name)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Could not restart '%s': %s", service_name, e.stderr.strip())
    except Exception as e:
        logger.error("Could not restart '%s': %s", service_name, e)
    return False


def check_and_deploy():
    """Checks origin/main for new commits and applies updates."""
    # 1. Fetch latest changes from GitHub
    ok, _, err = run_git(["fetch", "origin", "main"])
    if not ok:
        if "Could not resolve host" in err or "Network is unreachable" in err:
            logger.warning("Network offline or GitHub unreachable, will retry next cycle.")
        else:
            logger.warning("git fetch origin main warning: %s", err)
        return False

    # 2. Get local HEAD and remote origin/main hashes
    ok_local, local_hash, _ = run_git(["rev-parse", "HEAD"])
    ok_remote, remote_hash, _ = run_git(["rev-parse", "origin/main"])

    if not (ok_local and ok_remote):
        return False

    # 3. Check how many commits remote is ahead of local
    ok_count, count_str, _ = run_git(["rev-list", "--count", "HEAD..origin/main"])
    commits_behind = int(count_str) if (ok_count and count_str.isdigit()) else 0

    if commits_behind == 0 and local_hash == remote_hash:
        # Everything up to date
        return False

    if commits_behind == 0 and local_hash != remote_hash:
        # Local is ahead of remote (e.g. local unpushed commit)
        return False

    # 4. New commits detected on GitHub!
    logger.info("=" * 60)
    logger.info("🚀 [AUTO-DEPLOY] New commit(s) detected on GitHub origin/main!")
    logger.info("   Local commit : %s", local_hash[:8])
    logger.info("   Remote commit: %s (%d commit(s) behind)", remote_hash[:8], commits_behind)
    logger.info("=" * 60)

    # Find which files changed between local and remote
    _, diff_files_str, _ = run_git(["diff", "--name-only", "HEAD", "origin/main"])
    changed_files = [f for f in diff_files_str.split("\n") if f]
    logger.info("Changed files (%d): %s", len(changed_files), ", ".join(changed_files[:10]))

    # 5. Pull changes (Vercel-style fast-forward / reset)
    ok_pull, pull_out, pull_err = run_git(["reset", "--hard", "origin/main"])
    if not ok_pull:
        logger.error("git reset --hard origin/main failed: %s", pull_err)
        return False

    logger.info("Successfully updated working tree to origin/main commit %s", remote_hash[:8])

    # Also sync secondary folder if present
    if os.path.exists(SECONDARY_DIR) and SECONDARY_DIR != REPO_DIR:
        try:
            subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=SECONDARY_DIR, capture_output=True)
            logger.info("Synced secondary directory: %s", SECONDARY_DIR)
        except Exception as e:
            logger.warning("Could not sync secondary dir: %s", e)

    # 6. Check what services need reloading
    has_dep_changes = any("requirements" in f for f in changed_files)
    has_app_changes = any("Frontend/app.py" in f or "mongodb.py" in f or ".env" in f for f in changed_files)
    has_camera_changes = any(
        (f.endswith(".py") and ("/" not in f or f == "Frontend/daily_report_email.py"))
        or f in {".env", "lines_config.json"}
        or f.startswith("models/")
        or "requirements" in f
        for f in changed_files
    )
    has_frontend_only = all(
        f.startswith("Frontend/html/") or f.startswith("Frontend/assets/") or f.startswith("Frontend/public/") or f.endswith(".md")
        for f in changed_files
    )

    # Install new dependencies if requirements.txt changed
    if has_dep_changes and os.path.exists(VENV_PIP):
        logger.info("📦 Dependencies changed. Running pip install...")
        subprocess.run([VENV_PIP, "install", "-r", "requirements.txt"], cwd=REPO_DIR)

    # Reload Flask web app if app.py changed
    if has_app_changes:
        logger.info("🔄 Reloading Flask web application (Frontend/app.py)...")
        app_script = os.path.join(REPO_DIR, "Frontend", "app.py")
        restart_process_by_pattern("Frontend/app.py", [VENV_PYTHON, app_script])

    # Reload camera AI engine if camera.py changed
    if has_camera_changes:
        logger.info("🎥 Reloading camera engine through inoutx-camera.service...")
        restart_systemd_service("inoutx-camera.service")

    if has_frontend_only:
        logger.info("⚡ Frontend static changes applied! Live on https://inoutx.zynix.us instantly (no restart needed).")

    # 7. Write deploy status
    _, commit_msg, _ = run_git(["log", "-1", "--pretty=%B"])
    _, commit_author, _ = run_git(["log", "-1", "--pretty=%an <%ae>"])

    status_data = {
        "status": "success",
        "last_deploy_time": datetime.now().isoformat(),
        "commit_hash": remote_hash,
        "commit_short": remote_hash[:8],
        "commit_message": commit_msg.strip(),
        "author": commit_author.strip(),
        "changed_files_count": len(changed_files),
        "changed_files": changed_files
    }

    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logger.warning("Could not write status file: %s", e)

    logger.info("✅ Deployment completed successfully for commit %s!", remote_hash[:8])
    return True


def run_daemon():
    logger.info("=" * 65)
    logger.info("IN/OUT X - Git Auto-Sync Poller Daemon Started")
    logger.info("Watching repository: %s", REPO_DIR)
    logger.info("Polling interval   : %d seconds", POLL_INTERVAL_SECONDS)
    logger.info("Deployment domain  : https://inoutx.zynix.us & https://camera.zynix.us")
    logger.info("Log file location  : %s", LOG_FILE)
    logger.info("=" * 65)

    while running:
        try:
            check_and_deploy()
        except Exception as e:
            logger.error("Error in auto-deploy loop: %s", e, exc_info=True)

        for _ in range(POLL_INTERVAL_SECONDS):
            if not running:
                break
            time.sleep(1)

    logger.info("Auto-deploy daemon stopped.")


if __name__ == "__main__":
    if "--check-now" in sys.argv:
        logger.info("Running single on-demand check...")
        check_and_deploy()
    elif "--status" in sys.argv:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                print(f.read())
        else:
            print("No deployment recorded yet.")
    else:
        run_daemon()
