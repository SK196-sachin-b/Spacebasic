

import os
import json
import boto3
from retrieval import retrieval_service
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# RAG FUNCTION (FIXED)
# ──────────────────────────────────────────────
def RAG_QA(query: str) -> str:
    print("\n" + "="*100)
    print(f"🔍 RAG_QA CALLED")
    print(f"📝 QUERY: {query}")
    print("="*100)

    try:
        print("🔄 Calling retrieval_service.search()...")
        results = retrieval_service.search(query, top_k=7)

        if not results:
            print("⚠️ No results → sending empty context")
            return "No relevant document context available."

        # ✅ DETAILED CHUNK DEBUGGING - Show what's being sent to LLM
        print("\n" + "🔍"*50)
        print("📦 CHUNKS SENT TO LLM - DETAILED DEBUG")
        print("🔍"*50)
        
        for i, r in enumerate(results):
            print(f"\n📄 CHUNK {i+1}:")
            print(f"   🆔 ID: {r.get('id', 'N/A')}")
            print(f"   📊 Hybrid Score: {r.get('hybrid_score', 0.0):.4f}")
            print(f"   🧠 Semantic Score: {r.get('semantic_score', 0.0):.4f}")
            print(f"   📝 BM25 Score: {r.get('bm25_score', 0.0):.4f}")
            print(f"   📁 Source File: {r.get('source_file', 'N/A')}")
            print(f"   📂 Folder: {r.get('folder', 'N/A')}")
            print(f"   🔢 Version: {r.get('version', 'N/A')}")
            print(f"   ✅ Is Active: {r.get('is_active', 'NOT_AVAILABLE')}")
            print(f"   📄 Page: {r.get('page_number', 'N/A')}")
            print(f"   📝 Content Preview: {r.get('content', '')[:100]}...")
            print("   " + "-"*60)

        # 🔥 FIX: DO NOT RETURN EARLY — only mark low confidence
        top_score = results[0].get("hybrid_score", 0.0)
        print(f"\n🔍 Top hybrid score: {top_score}")

        if top_score < 0.015:
            print("⚠️ Low relevance → using empty context instead of stopping")
            return "No relevant document context available."

    except Exception as e:
        print(f"❌ ERROR in RAG_QA: {e}")
        import traceback
        traceback.print_exc()
        return "No relevant document context available."

    # ──────────────────────────────────────────────
    # BUILD CONTEXT
    # ──────────────────────────────────────────────
    context_parts = []

    for i, r in enumerate(results):
        score = r.get("hybrid_score", 0.0)

        chunk_context = (
            f"[Rank {i+1} | Score: {score:.4f} | "
            f"Source: {r['source_file']} | "
            f"Category: {r['folder']} | "
            f"ID: {r.get('id', 'N/A')} | "
            f"Version: {r.get('version', 'N/A')} | "
            f"Active: {r.get('is_active', 'N/A')}]\n"
            f"{r['content']}"
        )
        context_parts.append(chunk_context)

    context = "\n\n---\n\n".join(context_parts)

    print(f"\n📦 FINAL CONTEXT LENGTH: {len(context)} chars")
    print("="*100)

    return context


# ──────────────────────────────────────────────
# QA SERVICE
# ──────────────────────────────────────────────
class QAService:

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

        self.model_id = "qwen.qwen3-vl-235b-a22b"  # ✅ Changed to Qwen as requested
        print("model--", self.model_id)

        # 🔥 SYSTEM PROMPT (FINAL VERSION)
        self.system_prompt = """
You are a smart AI assistant for SpaceBasic university.

You must behave in THREE modes:

-----------------------------------
MODE 1: GENERAL CONVERSATION
-----------------------------------
If the user input is:
- greeting (hi, hello, hey)
- casual talk (how are you, thanks, what can you do)

Then:
- Respond like a friendly chatbot
- DO NOT use context
- DO NOT say fallback message

-----------------------------------
MODE 2: DOCUMENT-BASED QA
-----------------------------------
If the question is related to university documents:
- Use ONLY the provided context
- DO NOT hallucinate
- Combine chunks if needed

-----------------------------------
MODE 3: UNKNOWN / OUT-OF-SCOPE
-----------------------------------
If:
- The question is NOT a greeting
- AND the context does NOT contain relevant info

Then:
- DO NOT answer using your own knowledge
- Respond ONLY with:

"I couldn't find enough information about this in the available documents."

-----------------------------------
IMPORTANT RULE:
- DO NOT use your own knowledge for factual questions
- ONLY greetings are allowed outside documents
"""

    # ──────────────────────────────────────────────
    # MAIN ASK FUNCTION
    # ──────────────────────────────────────────────
    def ask(self, question: str, session_id: str = None) -> str:
        try:
            print(f"\n{'='*100}")
            print(f"🤖 QA SERVICE (Qwen) - Using Bedrock Session Memory")
            print(f"❓ Question: {question}")
            if session_id:
                print(f"🔗 Session ID: {session_id}")
            print(f"{'='*100}")

            # ✅ STEP 1: Store messages in DB (for UI and audit only)
            if session_id:
                from db import db
                
                if db.connect():
                    # Ensure session exists in database
                    db.cursor.execute("SELECT session_id FROM chat_sessions WHERE session_id = %s;", (session_id,))
                    session_exists = db.cursor.fetchone()
                    
                    if not session_exists:
                        # Create the session if it doesn't exist
                        db.cursor.execute("INSERT INTO chat_sessions (session_id) VALUES (%s);", (session_id,))
                        db.connection.commit()
                        print(f"✅ Created missing session in QA: {session_id}")
                    
                    # Store user message (for UI display and audit)
                    db.store_message(session_id, "user", question)
                    db.close()

            # STEP 2: GET CONTEXT FOR DOCUMENT QUERIES
            print("🔄 Getting context from RAG...")
            context = RAG_QA(question)

            print("✅ Context ready")
            print(f"� Context length: {len(context)} characters")

            # STEP 3: BUILD SIMPLIFIED PROMPT (Bedrock handles conversation history)
            prompt = f"""Answer the user's question using the provided document context.

Rules:
- Use ONLY the context for factual answers
- Do NOT hallucinate
- If not found → say fallback message
- Conversation history is already handled by Bedrock

Question: {question}

Context:
{context}

Answer:"""

            print("🔄 Calling LLM with Bedrock session memory...")
            print(f"📝 Prompt length: {len(prompt)} characters")

            # STEP 4: CALL MODEL WITH SESSION ID (Bedrock native memory)
            body_data = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024
            }
            
            # 🔥 ADD SESSION ID FOR BEDROCK MEMORY
            if session_id:
                body_data["sessionId"] = session_id
                print(f"🧠 Using Bedrock session memory: {session_id}")

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body_data)
            )

            result = json.loads(response["body"].read())
            print("✅ LLM responded")

            # STEP 5: EXTRACT ANSWER
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()

                print(f"📝 Answer length: {len(answer)} characters")

                # 🔥 PRINT RESPONSE
                print("\n🤖 LLM RESPONSE:")
                print("-" * 80)
                print(answer)
                print("-" * 80)

                # ✅ STEP 6: Store assistant response in DB (for UI display and audit)
                if session_id:
                    from db import db
                    if db.connect():
                        # Ensure session still exists before storing response
                        db.cursor.execute("SELECT session_id FROM chat_sessions WHERE session_id = %s;", (session_id,))
                        session_exists = db.cursor.fetchone()
                        
                        if not session_exists:
                            # Create the session if it doesn't exist
                            db.cursor.execute("INSERT INTO chat_sessions (session_id) VALUES (%s);", (session_id,))
                            db.connection.commit()
                            print(f"✅ Created missing session for response: {session_id}")
                        
                        db.store_message(session_id, "assistant", answer)
                        db.close()

                return answer

            else:
                print("❌ Unexpected response format")
                return "I encountered an error processing your question."

        except Exception as e:
            print(f"❌ QA Service error: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error. Please try again."


# GLOBAL INSTANCE
qa_service = QAService()