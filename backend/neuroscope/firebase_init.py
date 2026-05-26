"""Firebase Admin SDK initialization and client providers.

Provides unified, safe initialization for storage, server, and seed scripts.
"""
from __future__ import annotations

import json
import logging
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)

_app = None
_db = None


def get_firebase_app():
    global _app
    if _app is not None:
        return _app

    # Check if already initialized by firebase_admin internally
    try:
        _app = firebase_admin.get_app()
        return _app
    except ValueError:
        pass

    # Load credentials
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
        # Check relative to backend/ root or absolute path
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
        logger.warning("No Firebase credentials provided. Falling back to default credentials.")
        try:
            cred = credentials.ApplicationDefault()
        except Exception as e:
            logger.error("Failed to load ApplicationDefault credentials: %s", e)

    # Initialize app
    options = {}
    if bucket_name:
        options["storageBucket"] = bucket_name

    _app = firebase_admin.initialize_app(cred, options)
    logger.info("Firebase Admin SDK successfully initialized.")
    return _app


def get_db():
    global _db
    if _db is None:
        get_firebase_app()
        _db = firestore.client()
    return _db


def get_bucket():
    get_firebase_app()
    return storage.bucket()
