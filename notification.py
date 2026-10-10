from datetime import datetime
from mongodb import alerts


def create_notification(
    title,
    message,
    notification_type="info",
    college=None,
    camera_id=None
):
    """
    Create a notification in MongoDB with camera and college metadata.
    """
    notification = {
        "title": str(title),
        "message": str(message),
        "type": str(notification_type),
        "created_at": datetime.now(),
        "read": False
    }

    if college:
        notification["college"] = college
    if camera_id:
        notification["camera_id"] = camera_id

    try:
        result = alerts.insert_one(notification)
        print("Notification created:", title)
        return str(result.inserted_id)
    except Exception as e:
        print("Notification error:", e)
        return None