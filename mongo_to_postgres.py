import argparse
import os
from datetime import date, datetime
from urllib.parse import quote_plus

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

from postgres import PostgresDatabase


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _source_uri():
    uri = os.getenv("MONGODB_URI")
    if uri:
        return uri

    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set MONGODB_URI, or provide the legacy MONGODB_USERNAME and "
            "MONGODB_PASSWORD variables for this one-time migration."
        )

    host = os.getenv("MONGODB_HOST", "cluster0.c2phmz2.mongodb.net")
    return (
        f"mongodb+srv://{quote_plus(username)}:{quote_plus(password)}"
        f"@{host}/?appName=INOUTX-migration"
    )


def _normalize(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if value.__class__.__name__ == "Decimal128":
        return str(value)
    return value


def _chunks(cursor, size):
    batch = []
    for document in cursor:
        batch.append(_normalize(document))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Inventory MongoDB collections and migrate them to PostgreSQL JSONB."
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--apply",
        action="store_true",
        help="copy source documents into PostgreSQL (rerunnable; does not delete destination rows)",
    )
    operation.add_argument(
        "--verify",
        action="store_true",
        help="compare source and destination collection counts without writing data",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    mongo = MongoClient(_source_uri(), serverSelectionTimeoutMS=10000)
    source = mongo[os.getenv("MONGO_DB", "IN_OUTX")]

    try:
        mongo.admin.command("ping")
        names = sorted(name for name in source.list_collection_names() if not name.startswith("system."))
        inventory = [(name, source[name].count_documents({})) for name in names]
        print(f"MongoDB source connected; database contains {len(inventory)} collections.")
        for name, count in inventory:
            print(f"{name}: {count} documents")

        if not args.apply and not args.verify:
            print("Inventory only. Use --apply to migrate or --verify to compare counts.")
            return 0

        target = PostgresDatabase()
        target.command("ping")
        print(f"PostgreSQL destination connected: {target.name}")

        if args.verify:
            mismatches = []
            for name, source_count in inventory:
                destination_count = target[name].count_documents({})
                print(f"{name}: MongoDB={source_count}, PostgreSQL={destination_count}")
                if source_count != destination_count:
                    mismatches.append(name)
            extra_collections = sorted(set(target.list_collection_names()) - set(names))
            if extra_collections:
                print("PostgreSQL-only collections:", ", ".join(extra_collections))
            if mismatches or extra_collections:
                print("Count verification failed.")
                return 1
            print("Count verification passed for all source collections.")
            return 0

        migrated = 0
        for name, expected_count in inventory:
            destination = target[name]
            copied = 0
            for batch in _chunks(source[name].find({}).batch_size(args.batch_size), args.batch_size):
                destination.insert_many(batch, replace_existing=True)
                copied += len(batch)
                migrated += len(batch)
            print(f"{name}: copied {copied}/{expected_count} documents")
            destination_count = destination.count_documents({})
            if copied != expected_count or destination_count != expected_count:
                raise RuntimeError(f"Count mismatch or concurrent writes detected for collection {name!r}")
        print(f"Migration complete: {migrated} documents copied. Source MongoDB data was not modified.")
        return 0
    finally:
        mongo.close()


if __name__ == "__main__":
    raise SystemExit(main())
