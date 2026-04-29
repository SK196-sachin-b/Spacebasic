#!/usr/bin/env python3

import argparse
import threading
import sys
import os

# Add service folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from ingestion import ingestion_service
from qa import qa_service


# -------------------------------
# START UNIFIED CLEANUP SCHEDULER
# -------------------------------
def start_scheduler_once():
    if not hasattr(start_scheduler_once, "started"):
        # Start unified cleanup scheduler (handles both documents and chat)
        start_unified_cleanup()
        start_scheduler_once.started = True
        print("✅ Unified cleanup scheduler started (documents + chat)")


# -------------------------------
# INGEST
# -------------------------------
def ingest_documents(file_path=None):
    print("🚀 Starting ingestion...")

    if file_path:
        ingestion_service.ingest_single_pdf(file_path)
    else:
        ingestion_service.ingest_documents()


# -------------------------------
# QUERY MODE
# -------------------------------
def query_mode():
    print("🤖 RAG QA System - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 50)

    while True:
        try:
            question = input("\n💬 Ask a question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not question:
                print("⚠️ Please enter a question.")
                continue

            print("\n🔄 Processing...")
            answer = qa_service.ask(question)
            print(f"\n🤖 Answer: {answer}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Document QA System")
    parser.add_argument(
        "--mode",
        choices=["ingest", "query"],
        required=True,
        help="Mode: 'ingest' or 'query'"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF"
    )

    args = parser.parse_args()

    # 🔥 START SCHEDULER HERE
    start_scheduler_once()

    # 🔥 RUN MODES
    if args.mode == "ingest":
        ingest_documents(args.file)

    elif args.mode == "query":
        query_mode()


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()