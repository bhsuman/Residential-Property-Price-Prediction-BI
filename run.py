"""
Application Entry Point for Melbourne Property Valuation & Adaptive Business Intelligence Platform.
Usage:
    python run.py
"""

import os
from src.config import DB_PATH, PREPROCESSOR_PATH
from src.etl_pipeline import run_etl_pipeline
from src.ml_models import train_and_evaluate_models
from src.web_app import app, abi_engine


def main():
    if not os.path.exists(DB_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print("Initializing ETL Star Schema Data Warehouse & Training ML Models...")
        run_etl_pipeline()
        train_and_evaluate_models()
        abi_engine.reload_artifacts()

    port = int(os.environ.get("PORT", 5000))
    print(f"Launching Melbourne Property Valuation & ABI Platform at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()

