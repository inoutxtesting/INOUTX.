import hashlib
import json
import os
import re
import secrets
import select
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2.extras import Json
from psycopg2.pool import ThreadedConnectionPool


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

ASCENDING = 1
DESCENDING = -1
_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{24}$")
_FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
_ISO_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?$"
)


class ObjectId(str):
    def __new__(cls, value):
        value = str(value)
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("document id must be a 24-character hexadecimal string")
        return super().__new__(cls, value.lower())


class _WriteResult:
    def __init__(self, matched_count=0, modified_count=0, upserted_id=None, inserted_id=None):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id
        self.inserted_id = inserted_id


class _DeleteResult:
    def __init__(self, deleted_count=0):
        self.deleted_count = deleted_count


def _new_id():
    return f"{int(time.time()):08x}{secrets.token_hex(8)}"


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (ObjectId,)) or value.__class__.__name__ == "ObjectId":
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"Unsupported database value: {type(value).__name__}")


def _json_value(value):
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _decode_value(value):
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, str) and _ISO_DATETIME_PATTERN.fullmatch(value):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def _get_nested(document, path, missing=None):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return missing
        value = value[key]
    return value


def _set_nested(document, path, value):
    parts = path.split(".")
    target = document
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _unset_nested(document, path):
    parts = path.split(".")
    target = document
    for part in parts[:-1]:
        target = target.get(part)
        if not isinstance(target, dict):
            return
    target.pop(parts[-1], None)


def _field_sql(path):
    if path == "_id":
        path = "_id"
    if not _FIELD_PATTERN.fullmatch(path):
        raise ValueError(f"Unsupported document field path: {path}")
    parts = path.split(".")
    return "(data #> ARRAY[" + ", ".join("%s" for _ in parts) + "])::jsonb", parts


def _compile_filter(query):
    clauses = []
    params = []

    for field, condition in (query or {}).items():
        if field in ("$and", "$or"):
            nested = [_compile_filter(item) for item in condition]
            if not nested:
                clauses.append("TRUE" if field == "$and" else "FALSE")
                continue
            joiner = " AND " if field == "$and" else " OR "
            clauses.append("(" + joiner.join(sql for sql, _ in nested) + ")")
            for _, values in nested:
                params.extend(values)
            continue

        expression, path_params = _field_sql(field)
        if not isinstance(condition, dict) or not any(
            str(key).startswith("$") for key in condition
        ):
            clauses.append(f"({expression} = %s::jsonb OR {expression} @> %s::jsonb)")
            params.extend([*path_params, _json_value(condition), *path_params, _json_value(condition)])
            continue

        for operator, value in condition.items():
            if operator == "$exists":
                clauses.append(f"{expression} IS {'NOT ' if value else ''}NULL")
                params.extend(path_params)
            elif operator in ("$eq", "$ne"):
                comparison = f"{expression} = %s::jsonb"
                if operator == "$ne":
                    clauses.append(f"({expression} IS NULL OR NOT ({comparison}))")
                    params.extend([*path_params, *path_params, _json_value(value)])
                else:
                    clauses.append(comparison)
                    params.extend([*path_params, _json_value(value)])
            elif operator in ("$gt", "$gte", "$lt", "$lte"):
                comparator = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[operator]
                clauses.append(f"{expression} {comparator} %s::jsonb")
                params.extend([*path_params, _json_value(value)])
            elif operator in ("$in", "$nin"):
                matches = []
                for item in value:
                    matches.append(f"({expression} = %s::jsonb OR {expression} @> %s::jsonb)")
                    params.extend([*path_params, _json_value(item), *path_params, _json_value(item)])
                comparison = " OR ".join(matches) or "FALSE"
                if operator == "$nin":
                    clauses.append(f"NOT ({comparison})")
                else:
                    clauses.append(f"({comparison})")
            elif operator in ("$regex", "$options"):
                if operator == "$options":
                    continue
                text_expression = "(data #>> ARRAY[" + ", ".join("%s" for _ in path_params) + "])"
                regex_options = condition.get("$options", "")
                clauses.append(f"{text_expression} {'~*' if 'i' in regex_options else '~'} %s")
                params.extend([*path_params, value])
            else:
                raise ValueError(f"Unsupported document query operator: {operator}")

    return (" AND ".join(clauses) or "TRUE"), params


def _path_duplicate(path_params):
    return list(path_params)


def _apply_update(document, update, inserting=False):
    changes = update or {}
    if not any(str(key).startswith("$") for key in changes):
        identifier = document.get("_id")
        document.clear()
        document.update(changes)
        if identifier is not None:
            document["_id"] = identifier
        return

    for path, value in changes.get("$set", {}).items():
        _set_nested(document, path, value)
    if inserting:
        for path, value in changes.get("$setOnInsert", {}).items():
            _set_nested(document, path, value)
    for path, value in changes.get("$inc", {}).items():
        current = _get_nested(document, path, 0)
        _set_nested(document, path, current + value)
    for path in changes.get("$unset", {}):
        _unset_nested(document, path)


def _project(document, projection):
    if not projection:
        return document
    included = [key for key, value in projection.items() if value and key != "_id"]
    excluded = [key for key, value in projection.items() if not value]
    if included:
        result = {}
        for path in included:
            value = _get_nested(document, path)
            if value is not None:
                _set_nested(result, path, value)
        if projection.get("_id", 1) and "_id" in document:
            result["_id"] = document["_id"]
        return result
    result = dict(document)
    for path in excluded:
        _unset_nested(result, path)
    return result


class PostgresCursor:
    def __init__(self, collection, query=None, projection=None):
        self.collection = collection
        self.query = query or {}
        self.projection = projection
        self.ordering = []
        self.offset = 0
        self.row_limit = None

    def sort(self, key_or_list, direction=None):
        self.ordering = (
            [(key_or_list, direction or ASCENDING)]
            if isinstance(key_or_list, str)
            else list(key_or_list)
        )
        return self

    def skip(self, count):
        self.offset = max(0, int(count))
        return self

    def limit(self, count):
        self.row_limit = max(0, int(count))
        return self

    def __iter__(self):
        where, params = _compile_filter(self.query)
        sql = f"SELECT data FROM inoutx_documents WHERE collection = %s AND ({where})"
        params = [self.collection.name, *params]
        if self.ordering:
            terms = []
            for path, direction in self.ordering:
                expression, path_params = _field_sql(path)
                terms.append(f"{expression} {'DESC' if direction == DESCENDING else 'ASC'} NULLS LAST")
                params.extend(path_params)
            sql += " ORDER BY " + ", ".join(terms)
        if self.row_limit is not None:
            sql += " LIMIT %s"
            params.append(self.row_limit)
        if self.offset:
            sql += " OFFSET %s"
            params.append(self.offset)

        with self.collection.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        for (document,) in rows:
            yield _project(_decode_value(document), self.projection)


class _ChangeStream:
    def __init__(self, collection):
        self.collection = collection
        self.connection = None

    def __enter__(self):
        self.connection = self.collection.database.open_listener()
        with self.connection.cursor() as cursor:
            cursor.execute(f'LISTEN "{self.collection.channel}"')
            self.connection.commit()
        return self

    def __exit__(self, *_):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __iter__(self):
        while self.connection is not None:
            if select.select([self.connection], [], [], 1.0)[0]:
                self.connection.poll()
                while self.connection.notifies:
                    notification = self.connection.notifies.pop(0)
                    change = json.loads(notification.payload)
                    if change["operationType"] != "delete":
                        change["fullDocument"] = self.collection.find_one(
                            {"_id": change["documentKey"]["_id"]}
                        )
                    yield change


class PostgresCollection:
    def __init__(self, database, name):
        if not _FIELD_PATTERN.fullmatch(name):
            raise ValueError("Invalid collection name")
        self.database = database
        self.name = name
        self.channel = "inoutx_" + hashlib.sha1(name.encode()).hexdigest()[:32]

    def find(self, query=None, projection=None, **_):
        return PostgresCursor(self, query, projection)

    def find_one(self, query=None, projection=None, **_):
        return next(iter(self.find(query, projection).limit(1)), None)

    def count_documents(self, query):
        where, params = _compile_filter(query)
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM inoutx_documents WHERE collection = %s AND ({where})",
                    [self.name, *params],
                )
                return cursor.fetchone()[0]

    def insert_one(self, document):
        stored = dict(document)
        stored.setdefault("_id", _new_id())
        identifier = str(stored["_id"])
        stored["_id"] = identifier
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO inoutx_documents (collection, id, data) VALUES (%s, %s, %s)",
                    [self.name, identifier, Json(stored, dumps=_json_value)],
                )
                self.database.notify(connection, self, "insert", identifier)
        return _WriteResult(upserted_id=identifier, inserted_id=identifier)

    def insert_many(self, documents, replace_existing=False):
        ids = []
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                for document in documents:
                    stored = dict(document)
                    stored.setdefault("_id", _new_id())
                    identifier = str(stored["_id"])
                    stored["_id"] = identifier
                    sql = "INSERT INTO inoutx_documents (collection, id, data) VALUES (%s, %s, %s)"
                    if replace_existing:
                        sql += " ON CONFLICT (collection, id) DO UPDATE SET data = EXCLUDED.data"
                    cursor.execute(sql, [self.name, identifier, Json(stored, dumps=_json_value)])
                    ids.append(identifier)
        return ids

    def update_one(self, query, update, upsert=False):
        where, params = _compile_filter(query)
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id, data FROM inoutx_documents WHERE collection = %s AND ({where}) LIMIT 1 FOR UPDATE",
                    [self.name, *params],
                )
                row = cursor.fetchone()
                if row is None:
                    if not upsert:
                        return _WriteResult()
                    document = {
                        key: value for key, value in (query or {}).items()
                        if not key.startswith("$") and not isinstance(value, dict)
                    }
                    document["_id"] = _new_id()
                    _apply_update(document, update, inserting=True)
                    identifier = str(document["_id"])
                    document["_id"] = identifier
                    cursor.execute(
                        "INSERT INTO inoutx_documents (collection, id, data) VALUES (%s, %s, %s)",
                        [self.name, identifier, Json(document, dumps=_json_value)],
                    )
                    self.database.notify(connection, self, "insert", identifier)
                    return _WriteResult(upserted_id=identifier)

                identifier, original = row
                document = _decode_value(original)
                before = _json_value(document)
                _apply_update(document, update)
                document["_id"] = identifier
                after = _json_value(document)
                if before != after:
                    cursor.execute(
                        "UPDATE inoutx_documents SET data = %s WHERE collection = %s AND id = %s",
                        [Json(document, dumps=_json_value), self.name, identifier],
                    )
                    self.database.notify(connection, self, "update", identifier)
                return _WriteResult(matched_count=1, modified_count=int(before != after))

    def delete_one(self, query):
        where, params = _compile_filter(query)
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM inoutx_documents WHERE ctid = (SELECT ctid FROM inoutx_documents WHERE collection = %s AND ({where}) LIMIT 1) RETURNING id",
                    [self.name, *params],
                )
                row = cursor.fetchone()
                if row:
                    self.database.notify(connection, self, "delete", row[0])
        return _DeleteResult(int(row is not None))

    def delete_many(self, query):
        where, params = _compile_filter(query)
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM inoutx_documents WHERE collection = %s AND ({where}) RETURNING id",
                    [self.name, *params],
                )
                identifiers = [row[0] for row in cursor.fetchall()]
                for identifier in identifiers:
                    self.database.notify(connection, self, "delete", identifier)
        return _DeleteResult(len(identifiers))

    def create_index(self, keys, name=None, unique=False, **options):
        fields = [(keys, ASCENDING)] if isinstance(keys, str) else list(keys)
        if not fields:
            raise ValueError("At least one index field is required")
        index_name = name or f"inoutx_{self.name}_{'_'.join(key for key, _ in fields)}_idx"
        if not re.fullmatch(r"[A-Za-z0-9_]+", index_name):
            raise ValueError("Invalid index name")
        expressions = []
        for path, direction in fields:
            if not _FIELD_PATTERN.fullmatch(path):
                raise ValueError(f"Unsupported index field path: {path}")
            json_path = "{" + ",".join(path.split(".")) + "}"
            expressions.append(f"(data #> '{json_path}') {'DESC' if direction == DESCENDING else 'ASC'}")
        uniqueness = "UNIQUE " if unique else ""
        collection = self.name.replace("'", "''")
        sql = (
            f"CREATE {uniqueness}INDEX IF NOT EXISTS {index_name} "
            f"ON inoutx_documents (collection, {', '.join(expressions)}) "
            f"WHERE collection = '{collection}'"
        )
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
        return index_name

    def watch(self, *_args, **_kwargs):
        return _ChangeStream(self)


class PostgresDatabase:
    def __init__(self):
        self.dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        parsed = urlsplit(self.dsn) if self.dsn else None
        self.name = (parsed.path.strip("/") if parsed else "") or os.getenv("POSTGRES_DB", "inoutx")
        self._pool = None
        self._pool_lock = threading.Lock()
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def __getitem__(self, collection_name):
        return PostgresCollection(self, collection_name)

    def _connection_parameters(self):
        if not self.dsn:
            raise RuntimeError("DATABASE_URL is required for PostgreSQL")
        parsed = urlsplit(self.dsn)
        if parsed.scheme not in ("postgres", "postgresql"):
            raise RuntimeError("DATABASE_URL must use the postgres:// or postgresql:// scheme")
        if parsed.hostname not in (None, "localhost", "127.0.0.1", "::1"):
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query.setdefault("sslmode", "require")
            return urlunsplit(parsed._replace(query=urlencode(query)))
        return self.dsn

    def _get_pool(self):
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    maximum = max(2, int(os.getenv("POSTGRES_POOL_SIZE", "20")))
                    self._pool = ThreadedConnectionPool(1, maximum, self._connection_parameters())
        return self._pool

    @contextmanager
    def connection(self):
        pool = self._get_pool()
        connection = pool.getconn()
        try:
            self._ensure_schema(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            pool.putconn(connection, close=bool(connection.closed))

    def _ensure_schema(self, connection):
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS inoutx_documents ("
                        "collection TEXT NOT NULL, id TEXT NOT NULL, data JSONB NOT NULL, "
                        "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                        "PRIMARY KEY (collection, id))"
                    )
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS inoutx_documents_data_gin "
                        "ON inoutx_documents USING GIN (data jsonb_path_ops)"
                    )
                connection.commit()
                self._schema_ready = True
            except Exception:
                connection.rollback()
                raise

    def open_listener(self):
        return connect(self._connection_parameters())

    def notify(self, connection, collection, operation, identifier):
        payload = json.dumps({
            "operationType": operation,
            "documentKey": {"_id": identifier},
        }, separators=(",", ":"))
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_notify(%s, %s)", [collection.channel, payload])

    def command(self, command_name):
        if command_name != "ping":
            raise ValueError(f"Unsupported PostgreSQL command: {command_name}")
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return {"ok": 1, "server": "PostgreSQL"}

    def list_collection_names(self):
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT collection FROM inoutx_documents ORDER BY collection")
                return [row[0] for row in cursor.fetchall()]


db = PostgresDatabase()

users = db["users"]
email_verifications = db["email_verifications"]
password_resets = db["password_resets"]
college_buses = db["college_buses"]
bus_logs = db["bus_logs"]
staff_vehicles = db["staff_vehicles"]
staff_logs = db["staff_logs"]
visitor_vehicles = db["visitor_vehicles"]
unknown_vehicles = visitor_vehicles
visitor_logs = db["visitor_logs"]
alerts = db["alerts"]
detections = db["detections"]
vehicle_states = db["vehicle_states"]
gate_events = db["gate_events"]
login_logs = db["login_logs"]
user_presence = db["user_presence"]


def clean_plate(plate):
    if plate is None:
        return ""
    return str(plate).upper().replace(" ", "").replace("-", "").replace(".", "").replace("_", "").strip()


def clean_bus_number(bus_no):
    if bus_no is None:
        return ""
    value = str(bus_no).strip()
    return "" if value.lower() in {"", "unknown", "none", "null"} else value


def find_college_bus(plate):
    return college_buses.find_one({"plate": clean_plate(plate), "status": "ACTIVE"}) if plate else None


def find_bus_by_number(bus_no):
    number = clean_bus_number(bus_no)
    return college_buses.find_one({"bus_no": number, "status": "ACTIVE"}) or college_buses.find_one(
        {"bus_number": number, "status": "ACTIVE"}
    ) if number else None


def find_best_matching_bus(ocr_plate, threshold=80):
    from rapidfuzz import fuzz, process

    plate = clean_plate(ocr_plate)
    buses = list(college_buses.find({"status": "ACTIVE"})) if plate else []
    plate_map = {clean_plate(bus.get("plate")): bus for bus in buses if clean_plate(bus.get("plate"))}
    match = process.extractOne(plate, list(plate_map), scorer=fuzz.ratio) if plate_map else None
    return plate_map[match[0]] if match and match[1] >= threshold else None


def find_staff_vehicle(plate):
    return staff_vehicles.find_one({"plate": clean_plate(plate), "status": "ACTIVE"}) if plate else None


def verify_college_bus(plate):
    bus = find_college_bus(plate)
    if bus:
        return {"category": "BUS", "verified": True, "data": bus}
    staff = find_staff_vehicle(plate)
    if staff:
        return {"category": "STAFF", "verified": True, "data": staff}
    return {"category": "UNKNOWN", "verified": False, "data": None}


def insert_gate_event(data):
    return gate_events.insert_one(data).upserted_id


def insert_bus_log(data):
    return bus_logs.insert_one(data).upserted_id


def save_staff_vehicle(data):
    return staff_logs.insert_one(data).upserted_id


def save_unknown_vehicle(data):
    return unknown_vehicles.insert_one(data).upserted_id


def get_vehicle_state(vehicle_key):
    return vehicle_states.find_one({"vehicle_key": vehicle_key}) if vehicle_key else None


def get_current_direction(vehicle_key):
    state = get_vehicle_state(vehicle_key)
    return state.get("current_status") if state else None


def calculate_gate2_direction(vehicle_key, first_direction="ENTRY"):
    status = get_current_direction(vehicle_key)
    return "EXIT" if status == "INSIDE" else "ENTRY" if status == "OUTSIDE" else first_direction


def update_vehicle_state(vehicle_key, category, plate, bus_number, direction, gate, camera_id, event_id=None):
    if not vehicle_key:
        return None
    now = datetime.now()
    status = "INSIDE" if direction == "ENTRY" else "OUTSIDE" if direction == "EXIT" else "UNKNOWN"
    return vehicle_states.update_one(
        {"vehicle_key": vehicle_key},
        {
            "$set": {
                "vehicle_key": vehicle_key, "category": category, "plate": clean_plate(plate),
                "bus_no": clean_bus_number(bus_number), "current_status": status,
                "last_direction": direction, "last_gate": gate, "last_camera": camera_id,
                "last_event_id": event_id, "last_seen": now,
            },
            "$setOnInsert": {"created_at": now},
            "$inc": {"crossing_count": 1},
        },
        upsert=True,
    )


def save_vehicle_movement(*, category, plate, bus_number, direction, gate, camera_id, vehicle_type,
                          track_id, image_path, college=None, vehicle_data=None):
    plate = clean_plate(plate)
    bus_number = clean_bus_number(bus_number)
    college = college or ("KRCE" if "KRCE" in f"{camera_id} {gate}".upper() else "KRCT")
    now = datetime.now()
    vehicle_key = f"PLATE:{plate}" if plate else f"BUS:{bus_number}" if category == "BUS" and bus_number else f"{gate}:{camera_id}:TRACK:{track_id}"
    event = {
        "vehicle_key": vehicle_key, "category": category, "plate": plate, "bus_no": bus_number,
        "direction": direction, "gate": gate, "camera_id": camera_id, "vehicle_type": vehicle_type,
        "track_id": track_id, "image_path": image_path, "college": college, "timestamp": now,
    }
    if vehicle_data is not None:
        event["vehicle_data"] = vehicle_data
    insert_gate_event(event)
    update_vehicle_state(vehicle_key, category, plate, bus_number, direction, gate, camera_id, event.get("_id"))
    return event


def recent_event_exists(vehicle_key, gate, camera_id, cooldown_seconds=20):
    cutoff = datetime.now().timestamp() - cooldown_seconds
    return gate_events.find_one({
        "vehicle_key": vehicle_key, "gate": gate, "camera_id": camera_id,
        "timestamp": {"$gte": datetime.fromtimestamp(cutoff)},
    }) is not None if vehicle_key else False


def get_all_logs():
    return list(gate_events.find().sort("timestamp", DESCENDING))


def get_all_bus_logs():
    return list(bus_logs.find().sort("timestamp", DESCENDING))


def get_all_staff_logs():
    return list(staff_logs.find().sort("timestamp", DESCENDING))


def get_all_unknown_vehicles():
    return list(unknown_vehicles.find().sort("timestamp", DESCENDING))


def get_all_college_buses():
    return list(college_buses.find().sort("bus_no", ASCENDING))


def get_active_college_buses():
    return list(college_buses.find({"status": "ACTIVE"}).sort("bus_no", ASCENDING))


def get_all_staff_vehicles():
    return list(staff_vehicles.find().sort("_id", DESCENDING))


def get_all_detections():
    return list(detections.find().sort("_id", DESCENDING))


def get_dashboard_stats():
    return {
        "total_buses": college_buses.count_documents({"status": "ACTIVE"}),
        "total_bus_logs": bus_logs.count_documents({}),
        "total_staff_vehicles": staff_vehicles.count_documents({"status": "ACTIVE"}),
        "total_unknown_vehicles": unknown_vehicles.count_documents({}),
        "total_gate_events": gate_events.count_documents({}),
    }


def create_indexes():
    for collection, fields, options in (
        (college_buses, [("plate", ASCENDING)], {}),
        (college_buses, [("bus_no", ASCENDING)], {}),
        (staff_vehicles, [("plate", ASCENDING)], {}),
        (gate_events, [("vehicle_key", ASCENDING), ("timestamp", DESCENDING)], {}),
        (gate_events, [("gate", ASCENDING), ("camera_id", ASCENDING), ("timestamp", DESCENDING)], {}),
        (vehicle_states, [("vehicle_key", ASCENDING)], {"unique": True}),
    ):
        collection.create_index(fields, **options)


def test_connection():
    return db.command("ping")