import json
import unittest
from datetime import datetime

from postgres import ObjectId, _apply_update, _compile_filter, _decode_value, _json_value


class PostgresAdapterTests(unittest.TestCase):
    def test_application_filter_operators_bind_every_parameter(self):
        filters = (
            {"email": "driver@example.test"},
            {"_id": {"$ne": "0123456789abcdef01234567"}},
            {"last_seen_at": {"$gte": datetime(2026, 10, 14, 10, 0)}},
            {"image": {"$exists": True, "$ne": ""}},
            {"status": {"$in": ["ACTIVE", "PENDING"]}},
            {"status": {"$in": []}},
            {"status": {"$nin": []}},
            {"email": {"$regex": "@example\\.test$", "$options": "i"}},
            {"$or": [{"role": "admin"}, {"active": True}]},
        )

        for document_filter in filters:
            with self.subTest(document_filter=document_filter):
                sql, parameters = _compile_filter(document_filter)
                self.assertEqual(sql.count("%s"), len(parameters))

    def test_update_operators_support_nested_fields_and_upserts(self):
        document = {"profile": {"name": "Alex"}, "attempts": 1}
        _apply_update(
            document,
            {
                "$set": {"profile.active": True},
                "$setOnInsert": {"created_at": datetime(2026, 10, 14)},
                "$inc": {"attempts": 1},
                "$unset": {"profile.name": ""},
            },
            inserting=True,
        )

        self.assertEqual(document["profile"], {"active": True})
        self.assertEqual(document["attempts"], 2)
        self.assertEqual(document["created_at"], datetime(2026, 10, 14))

    def test_json_round_trip_preserves_ids_and_datetimes(self):
        document = {
            "_id": ObjectId("0123456789ABCDEF01234567"),
            "created_at": datetime(2026, 10, 14, 10, 30),
        }

        decoded = _decode_value(json.loads(_json_value(document)))

        self.assertEqual(decoded["_id"], "0123456789abcdef01234567")
        self.assertEqual(decoded["created_at"], document["created_at"])


if __name__ == "__main__":
    unittest.main()