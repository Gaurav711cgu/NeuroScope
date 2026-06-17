"""Firebase Admin SDK initialization and client providers.

Provides unified, safe initialization for storage, server, and seed scripts.
Falls back to a persistent JSON-based mock database when credentials are not found.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)

_app = None
_db = None
_use_mock = False


# ─── Mock Firestore Database Client ──────────────────────────────────────────

class MockDocumentSnapshot:
    def __init__(self, doc_id: str, data: dict | None):
        self.id = doc_id
        self._data = data
        self.exists = data is not None

    def to_dict(self) -> dict:
        return self._data or {}


class MockDocumentReference:
    def __init__(self, collection_name: str, doc_id: str, db: MockFirestoreClient):
        self.collection_name = collection_name
        self.id = doc_id
        self.db = db

    def get(self) -> MockDocumentSnapshot:
        data = self.db._get_doc(self.collection_name, self.id)
        return MockDocumentSnapshot(self.id, data)

    def set(self, data: dict):
        self.db._set_doc(self.collection_name, self.id, data)

    def update(self, data: dict):
        self.db._update_doc(self.collection_name, self.id, data)

    def delete(self):
        self.db._delete_doc(self.collection_name, self.id)


class MockQuery:
    def __init__(self, collection_name: str, db: MockFirestoreClient, filters=None, orders=None, limit_val=None):
        self.collection_name = collection_name
        self.db = db
        self.filters = filters or []
        self.orders = orders or []
        self.limit_val = limit_val

    def where(self, field: str, op: str, value) -> MockQuery:
        return MockQuery(
            self.collection_name,
            self.db,
            self.filters + [(field, op, value)],
            self.orders,
            self.limit_val
        )

    def order_by(self, field: str, direction=None) -> MockQuery:
        return MockQuery(
            self.collection_name,
            self.db,
            self.filters,
            self.orders + [(field, direction)],
            self.limit_val
        )

    def limit(self, val: int) -> MockQuery:
        return MockQuery(
            self.collection_name,
            self.db,
            self.filters,
            self.orders,
            val
        )

    def _execute(self) -> list[MockDocumentSnapshot]:
        docs = self.db._get_collection(self.collection_name)
        filtered_docs = []
        for doc_id, doc_data in docs.items():
            match = True
            for field, op, value in self.filters:
                doc_val = doc_data.get(field)
                if op == "==":
                    if doc_val != value:
                        match = False
                elif op == "in":
                    if not isinstance(value, (list, tuple, set)) or doc_val not in value:
                        match = False
            if match:
                filtered_docs.append((doc_id, doc_data))

        # Apply ordering
        for field, direction in self.orders:
            reverse = False
            if direction and ("desc" in str(direction).lower() or direction == "DESCENDING"):
                reverse = True
            
            def sort_key(item):
                val = item[1].get(field)
                if val is None:
                    return (False, "") if isinstance(val, str) else (False, 0)
                return (True, val)
            
            filtered_docs.sort(key=sort_key, reverse=reverse)

        # Apply limit
        if self.limit_val is not None:
            filtered_docs = filtered_docs[:self.limit_val]

        return [MockDocumentSnapshot(d_id, d_data) for d_id, d_data in filtered_docs]

    def get(self) -> list[MockDocumentSnapshot]:
        return self._execute()

    def stream(self) -> list[MockDocumentSnapshot]:
        return self._execute()


class MockCollectionReference(MockQuery):
    def __init__(self, collection_name: str, db: MockFirestoreClient):
        super().__init__(collection_name, db)

    def document(self, doc_id: str) -> MockDocumentReference:
        return MockDocumentReference(self.collection_name, doc_id, self.db)


class MockFirestoreClient:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._data = {}
        self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error("Error saving mock db: %s", e)

    def _get_collection(self, name: str) -> dict:
        return self._data.setdefault(name, {})

    def _get_doc(self, coll: str, doc_id: str) -> dict | None:
        return self._get_collection(coll).get(doc_id)

    def _set_doc(self, coll: str, doc_id: str, data: dict):
        self._get_collection(coll)[doc_id] = data
        self._save()

    def _update_doc(self, coll: str, doc_id: str, data: dict):
        doc = self._get_collection(coll).setdefault(doc_id, {})
        doc.update(data)
        self._save()

    def _delete_doc(self, coll: str, doc_id: str):
        collection = self._get_collection(coll)
        if doc_id in collection:
            del collection[doc_id]
            self._save()

    def collection(self, name: str) -> MockCollectionReference:
        return MockCollectionReference(name, self)


class MockBlob:
    def __init__(self, path: str):
        self.path = path

    def upload_from_string(self, data, content_type=None):
        pass

    def download_as_bytes(self) -> bytes:
        raise FileNotFoundError("Mock blob not found")


class MockBucket:
    def blob(self, path: str) -> MockBlob:
        return MockBlob(path)


# ─── Initialization logic ───────────────────────────────────────────────────

def get_firebase_app():
    global _app, _use_mock
    if _app is not None:
        return _app

    try:
        _app = firebase_admin.get_app()
        return _app
    except ValueError:
        pass

    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET")

    cred = None
    if service_account_json:
        try:
            logger.info("Initializing Firebase with raw JSON credentials")
            cred_dict = json.loads(service_account_json)
            cred = credentials.Certificate(cred_dict)
        except Exception as e:
            logger.error("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)

    if cred is None and service_account_path:
        paths_to_check = [
            service_account_path,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), service_account_path)
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    logger.info("Initializing Firebase with credential path: %s", p)
                    cred = credentials.Certificate(p)
                    break
                except Exception as e:
                    logger.error("Failed to load credential file %s: %s", p, e)

    if cred is None:
        logger.warning("No Firebase credentials provided. Attempting default credentials.")
        try:
            cred = credentials.ApplicationDefault()
        except Exception as e:
            logger.warning("Failed to load ApplicationDefault credentials: %s. Switching to Mock DB mode.", e)
            _use_mock = True
            return None

    try:
        options = {}
        if bucket_name:
            options["storageBucket"] = bucket_name
        _app = firebase_admin.initialize_app(cred, options)
        logger.info("Firebase Admin SDK successfully initialized.")
        return _app
    except Exception as e:
        logger.warning("Firebase initialization failed: %s. Switching to Mock DB mode.", e)
        _use_mock = True
        return None


def get_db():
    global _db, _use_mock
    if _db is None:
        get_firebase_app()
        if _use_mock:
            mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock_firestore.json")
            logger.info("Using persistent local mock database at: %s", mock_path)
            _db = MockFirestoreClient(mock_path)
        else:
            try:
                _db = firestore.client()
            except Exception as e:
                logger.warning("Failed to get Firestore client: %s. Falling back to mock DB.", e)
                _use_mock = True
                mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock_firestore.json")
                _db = MockFirestoreClient(mock_path)
    return _db


def get_bucket():
    global _use_mock
    if _use_mock:
        return MockBucket()
    try:
        get_firebase_app()
        return storage.bucket()
    except Exception:
        logger.warning("Failed to get storage bucket, returning mock bucket.")
        return MockBucket()
