from postgres import email_verifications, users

# -----------------------------
# USERS
# -----------------------------

user_count = users.count_documents({})
print(f"Users before deletion: {user_count}")

result = users.delete_many({})

print(f"Deleted {result.deleted_count} user accounts.")

# -----------------------------
# PENDING OTP VERIFICATIONS
# -----------------------------

otp_count = email_verifications.count_documents({})
print(f"Pending OTP records: {otp_count}")

otp_result = email_verifications.delete_many({})

print(f"Deleted {otp_result.deleted_count} pending OTP records.")

# -----------------------------
# VERIFY
# -----------------------------

print(f"Users remaining: {users.count_documents({})}")
print(
    f"OTP records remaining: "
    f"{email_verifications.count_documents({})}"
)