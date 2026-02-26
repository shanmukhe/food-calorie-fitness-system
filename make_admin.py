from db import db

db.execute(
    "UPDATE users SET is_admin=1 WHERE username=?",
    ("Shannu",)
)

print("Admin access granted.")