# ============================================================
# FILE: bedrock_reranker.py
# PURPOSE:
# Pass ALL retrieved chunks in ONE request to Bedrock model
# Model returns best top 7 chunk IDs
# No cropping / no truncation of chunk content
# ============================================================

import os
import json
import re
import boto3
from dotenv import load_dotenv

load_dotenv()


class BedrockReranker:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

        # use cheaper model for reranking
        self.model_id = "qwen.qwen3-vl-235b-a22b"
        # You can also use Nova Lite / Claude Haiku

    # --------------------------------------------------------
    # MAIN RERANK METHOD
    # --------------------------------------------------------
    def rerank(self, query, chunks, top_n=7):
        """
        query  = user query string
        chunks = list of dicts from retrieval.py
        returns top_n reranked chunks
        """

        if not chunks:
            return []

        print("\n🔄 BEDROCK RERANKER STARTED")
        print(f"📝 Query: {query}")
        print(f"📦 Total candidate chunks: {len(chunks)}")
        
        # 🔍 DETAILED INPUT CHUNKS DEBUG
        print("\n📦 CHUNKS SENT TO RERANKER:")
        print("-" * 80)
        for i, row in enumerate(chunks, start=1):
            content_preview = row.get('content', '')[:100] + "..." if len(row.get('content', '')) > 100 else row.get('content', '')
            hybrid_score = row.get('hybrid_score', 0.0)
            print(f"   Chunk {i}: ID={row.get('id')} | Hybrid={hybrid_score:.4f} | v{row.get('version', 1)}")
            print(f"   Content: {content_preview}")
            print(f"   File: {row.get('source_file', 'N/A')} | Folder: {row.get('folder', 'N/A')}")
            print("-" * 80)

        # ----------------------------------------------------
        # Build FULL prompt (NO CROPPING)
        # ----------------------------------------------------
        chunk_text = []

        for i, row in enumerate(chunks, start=1):
            chunk_text.append(
                f"""
CHUNK_ID: {i}
DB_ID: {row.get('id')}
SOURCE_FILE: {row.get('source_file')}
VERSION: {row.get('version')}
CONTENT:
{row.get('content')}
"""
            )

        joined_chunks = "\n\n=============================\n\n".join(chunk_text)

        prompt = f"""
You are an expert retrieval reranker.

Your task:
Given a user query and multiple document chunks,
select the BEST {top_n} chunks that together answer the query most completely.

IMPORTANT:
- Understand full meaning of query
- If query has multiple questions, choose chunks covering ALL parts
- Prefer latest versions when content is similar
- Avoid duplicates
- Return ONLY JSON list of chunk ids in best order

Example:
[3,1,7,2,5,4,6]

USER QUERY:
{query}

DOCUMENT CHUNKS:
{joined_chunks}

Return ONLY JSON array:
"""

        # ----------------------------------------------------
        # CALL MODEL ONCE
        # ----------------------------------------------------
        body = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())

        # ----------------------------------------------------
        # EXTRACT RESPONSE TEXT
        # ----------------------------------------------------
        if "choices" in result:
            text = result["choices"][0]["message"]["content"].strip()
        else:
            text = str(result)

        print("\n🤖 RAW RERANK RESPONSE:")
        print(text)
        
        # ----------------------------------------------------
        # Parse JSON IDs
        # ----------------------------------------------------
        try:
            ids = json.loads(text)
            print(f"\n✅ Successfully parsed JSON IDs: {ids}")

        except:
            # fallback parse numbers
            ids = list(map(int, re.findall(r"\d+", text)))
            print(f"\n⚠️ JSON parse failed, extracted numbers: {ids}")

        # keep only valid ids
        valid_ids = [x for x in ids if 1 <= x <= len(chunks)]
        invalid_ids = [x for x in ids if not (1 <= x <= len(chunks))]
        
        if invalid_ids:
            print(f"🚫 Invalid chunk IDs filtered out: {invalid_ids}")
        
        print(f"✅ Valid chunk IDs: {valid_ids}")

        # unique preserve order
        seen = set()
        final_ids = []

        for x in valid_ids:
            if x not in seen:
                seen.add(x)
                final_ids.append(x)

        final_ids = final_ids[:top_n]
        
        print(f"🎯 Final selected IDs (top {top_n}): {final_ids}")

        # fallback if model gives less ids
        if len(final_ids) < top_n:
            print(f"⚠️ Model returned {len(final_ids)} IDs, need {top_n}. Adding fallback chunks...")
            for i in range(1, len(chunks)+1):
                if i not in seen:
                    final_ids.append(i)
                    print(f"   Added fallback chunk ID: {i}")
                if len(final_ids) == top_n:
                    break

        # ----------------------------------------------------
        # Return selected chunks
        # ----------------------------------------------------
        reranked = [chunks[i-1] for i in final_ids]

        print("\n🏆 FINAL RERANKED CHUNKS:")
        print("-" * 80)
        for rank, row in enumerate(reranked, start=1):
            content_preview = row.get('content', '')[:100] + "..." if len(row.get('content', '')) > 100 else row.get('content', '')
            hybrid_score = row.get('hybrid_score', 0.0)
            print(f"   Rank {rank}: ID={row.get('id')} | Hybrid={hybrid_score:.4f} | v{row.get('version', 1)}")
            print(f"   Content: {content_preview}")
            print(f"   File: {row.get('source_file', 'N/A')} | Folder: {row.get('folder', 'N/A')}")
            print("-" * 80)
        
        print(f"✅ Reranker returned {len(reranked)} chunks (requested: {top_n})")

        return reranked


# global instance
reranker_service = BedrockReranker()