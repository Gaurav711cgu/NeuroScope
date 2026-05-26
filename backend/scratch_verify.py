import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify")

def verify_all():
    print("=== NeuroScope v2 Configuration Verification ===\n")
    
    # 1. Verify Firebase Setup
    print("1. Testing Firebase Connection...")
    try:
        from neuroscope.firebase_init import get_db
        db = get_db()
        # Try to write/read a temporary document
        doc_ref = db.collection("verify_test").document("status")
        doc_ref.set({"status": "OK", "timestamp": "checking"})
        data = doc_ref.get().to_dict()
        assert data and data["status"] == "OK", "Data readback failed"
        # Cleanup
        doc_ref.delete()
        print("   ✅ Firebase Firestore database connection works perfectly!\n")
    except Exception as e:
        print(f"   ❌ Firebase Firestore test failed: {e}\n")
        
    # 2. Verify LLM / Google Gemini API Key
    print("2. Testing Google Gemini API Connection...")
    try:
        from neuroscope.llm import _init_client
        import google.generativeai as genai
        import asyncio

        async def test_llm():
            _init_client()
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash"
            )
            response = await model.generate_content_async(
                "Hello, respond with 'Ready' if you can read this."
            )
            return response.text.strip()

        res = asyncio.run(test_llm())
        print(f"   Response: '{res}'")
        if "ready" in res.lower():
            print("   ✅ Google Gemini API key and gemini-1.5-flash model are active and working!\n")
        else:
            print("   ⚠️  Gemini returned unexpected response, but connection succeeded!\n")
    except Exception as e:
        print(f"   ❌ Google Gemini API test failed: {e}\n")

if __name__ == "__main__":
    verify_all()
