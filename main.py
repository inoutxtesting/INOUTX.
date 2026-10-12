from config import PROJECT_NAME
from camera import start_camera

print("=" * 50)
print(PROJECT_NAME)
print("AI Based College Transport Management System")
print("=" * 50)


import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from config import PROJECT_NAME
from camera import start_camera


def run_daily_email_report():
    """Generate and email the daily vehicle report."""
    try:
        from Frontend.daily_report_email import generate_and_send

        print("[DAILY REPORT] Starting scheduled report...")
        generate_and_send()
        print("[DAILY REPORT] Scheduled report completed.")
    except Exception:
        logging.exception("[DAILY REPORT] Report generation or email failed.")


def start_report_scheduler():
    """Run the daily email report at 9:30 AM India time."""
    scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Kolkata"))

    scheduler.add_job(
        run_daily_email_report,
        trigger="cron",
        hour=9,
        minute=30,
        id="inoutx_daily_email_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    print("[DAILY REPORT] Scheduler started: every day at 09:30 AM IST.")
    return scheduler


print("=" * 50)
print(PROJECT_NAME)
print("AI Based College Transport Management System")
print("=" * 50)

# Start the scheduler in the background before starting the cameras.
report_scheduler = start_report_scheduler()

# Preserve the existing dual-camera engine and calibration workflow.
start_camera()

# Start Dual-Camera Engine with Interactive Mouse-Based Line Calibration in Camera Preview
start_camera()