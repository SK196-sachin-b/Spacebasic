
# import concurrent.futures
# import numpy as np

# from db import db
# from embedding import embedding_service
# from langchain_community.vectorstores.utils import maximal_marginal_relevance


# class RetrievalService:

#     def __init__(self):
#         self.WEIGHT_SEMANTIC = 0.7
#         self.WEIGHT_BM25 = 0.3
#         self.HYBRID_THRESHOLD = 0.20

#     # ──────────────────────────────────────────────
#     # NORMALIZE BM25
#     # ──────────────────────────────────────────────
#     def _normalize(self, scores):
#         if not scores:
#             return {}

#         vals = list(scores.values())
#         min_v, max_v = min(vals), max(vals)

#         if max_v == min_v:
#             return {k: 1.0 for k in scores}

#         return {
#             k: (v - min_v) / (max_v - min_v)
#             for k, v in scores.items()
#         }

#     # ──────────────────────────────────────────────
#     # SEMANTIC SEARCH
#     # ──────────────────────────────────────────────
#     def _semantic_search(self, query_embedding, top_k):
#         try:
#             return db.search_similar(query_embedding, top_k=top_k)
#         except Exception as e:
#             print(f"❌ Semantic error: {e}")
#             return []

#     # ──────────────────────────────────────────────
#     # BM25 SEARCH
#     # ──────────────────────────────────────────────
#     def _bm25_search(self, query, top_k):
#         try:
#             return db.search_bm25(query, top_k=top_k)
#         except Exception as e:
#             print(f"❌ BM25 error: {e}")
#             return []

#     # ──────────────────────────────────────────────
#     # HYBRID FUSION (FIXED)
#     # ──────────────────────────────────────────────
#     def _fusion(self, semantic_results, bm25_results):

#         semantic_scores = {
#             r["id"]: float(r["similarity_score"])
#             for r in semantic_results
#         }

#         bm25_scores = {
#             r["id"]: float(r.get("bm25_score", 0.0))
#             for r in bm25_results
#         }

#         bm25_norm = self._normalize(bm25_scores)

#         all_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

#         id_to_chunk = {}

#         # 🔥 Merge results safely
#         for r in semantic_results + bm25_results:
#             doc_id = r["id"]

#             if doc_id not in id_to_chunk:
#                 id_to_chunk[doc_id] = dict(r)
#             else:
#                 existing = id_to_chunk[doc_id]

#                 # 🔥 ALWAYS preserve valid embedding
#                 if r.get("embedding") and (
#                     not existing.get("embedding") or len(existing.get("embedding", [])) == 0
#                 ):
#                     existing["embedding"] = r["embedding"]

#         # 🔥 EXTRA SAFETY: semantic embeddings always win
#         for r in semantic_results:
#             doc_id = r["id"]
#             if doc_id in id_to_chunk and r.get("embedding"):
#                 id_to_chunk[doc_id]["embedding"] = r["embedding"]

#         fused = []

#         for doc_id in all_ids:
#             chunk = id_to_chunk[doc_id]

#             s = semantic_scores.get(doc_id, 0.0)
#             b = bm25_norm.get(doc_id, 0.0)

#             chunk["semantic_score"] = s
#             chunk["bm25_score"] = b

#             chunk["hybrid_score"] = (
#                 self.WEIGHT_SEMANTIC * s +
#                 self.WEIGHT_BM25 * b
#             )

#             fused.append(chunk)

#         return sorted(fused, key=lambda x: x["hybrid_score"], reverse=True)

#     # ──────────────────────────────────────────────
#     # MMR (FINAL FIXED VERSION)
#     # ──────────────────────────────────────────────
#     def _apply_mmr(self, candidates, query_embedding, top_k):
#         if not candidates:
#             return []

#         valid = []

#         # 🔥 Enhanced embedding validation with detailed logging
#         print(f"\n🧪 MMR VALIDATION DETAILS:")
#         for i, c in enumerate(candidates[:3]):  # Check first 3 for debugging
#             emb = c.get("embedding")
#             print(f"  Candidate {i+1}: ID={c['id']}")
#             print(f"    - Embedding exists: {emb is not None}")
#             print(f"    - Embedding type: {type(emb)}")
#             print(f"    - Embedding length: {len(emb) if emb else 'N/A'}")
            
#             if emb and len(emb) > 0:
#                 print(f"    - First element type: {type(emb[0])}")
#                 print(f"    - First 3 values: {emb[:3] if len(emb) >= 3 else emb}")

#         # 🔥 Robust embedding validation and conversion
#         for c in candidates:
#             emb = c.get("embedding")
            
#             try:
#                 if emb is not None and len(emb) > 0:
#                     # Handle different embedding formats
#                     if isinstance(emb, str):
#                         # If it's a string, try to parse it
#                         import ast
#                         emb = ast.literal_eval(emb)
#                     elif hasattr(emb, 'tolist'):
#                         # If it's a numpy array, convert to list
#                         emb = emb.tolist()
                    
#                     # Convert all elements to float
#                     emb = [float(x) for x in emb]
                    
#                     # Validate embedding dimension (should be 1024 for Titan)
#                     if len(emb) == 1024:  # or whatever your expected dimension is
#                         c["embedding"] = emb
#                         valid.append(c)
#                         print(f"✅ Valid embedding for ID {c['id']}")
#                     else:
#                         print(f"❌ Wrong embedding dimension for ID {c['id']}: {len(emb)}")
#                 else:
#                     print(f"❌ No embedding for ID {c['id']}")
                    
#             except Exception as e:
#                 print(f"❌ Embedding conversion error for ID {c['id']}: {e}")
#                 continue

#         print(f"🔍 MMR Validation Summary: {len(valid)}/{len(candidates)} valid embeddings")

#         if not valid:
#             print("⚠️ No valid embeddings → fallback")
#             return candidates[:top_k]

#         if len(valid) <= top_k:
#             return valid

#         try:
#             print(f"🎯 Applying LangChain MMR to {len(valid)} candidates")
            
#             candidate_vecs = np.array(
#                 [c["embedding"] for c in valid],
#                 dtype=np.float32
#             )

#             query_vec = np.array(query_embedding, dtype=np.float32)

#             print(f"📊 MMR Input shapes: query={query_vec.shape}, candidates={candidate_vecs.shape}")

#             selected_indices = maximal_marginal_relevance(
#                 query_embedding=query_vec,
#                 embedding_list=candidate_vecs,
#                 lambda_mult=0.7,
#                 k=top_k
#             )

#             print(f"✅ MMR selected indices: {selected_indices}")
#             selected_results = [valid[i] for i in selected_indices]
            
#             print(f"🏆 MMR Success: Selected {len(selected_results)} diverse results")
#             return selected_results

#         except Exception as e:
#             print(f"❌ MMR failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return candidates[:top_k]

#     # ──────────────────────────────────────────────
#     # MAIN SEARCH
#     # ──────────────────────────────────────────────
#     def search(self, query, top_k=10, source_file=None):

#         if not db.ensure_connected():
#             print("❌ DB connection failed")
#             return []

#         print("\n" + "="*80)
#         print("🔎 HYBRID + MMR SEARCH")
#         print("="*80)
#         print(f"Query: {query}")

#         # STEP 1: Embedding
#         query_embedding = embedding_service.embed_text(query)

#         if query_embedding is None:
#             print("❌ Failed to generate embedding")
#             return []

#         fetch_k = min(top_k * 4, 50)

#         # STEP 2: Parallel search
#         with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#             sem_future = executor.submit(self._semantic_search, query_embedding, fetch_k)
#             bm25_future = executor.submit(self._bm25_search, query, fetch_k)

#             semantic_results = sem_future.result()
#             bm25_results = bm25_future.result()

#         if not semantic_results and not bm25_results:
#             return []

#         # STEP 3: Fusion
#         fused_results = self._fusion(semantic_results, bm25_results)

#         # STEP 4: Threshold
#         filtered = [
#             r for r in fused_results
#             if r["hybrid_score"] >= self.HYBRID_THRESHOLD
#         ]

#         if not filtered:
#             filtered = fused_results[:fetch_k]

#         # STEP 5: MMR
#         candidates = sorted(
#             filtered,
#             key=lambda x: x["hybrid_score"],
#             reverse=True
#         )

#         print("\n🧪 EMBEDDING CHECK BEFORE MMR")
#         for r in candidates[:5]:
#             print(f"ID={r['id']} →", "OK" if r.get("embedding") else "MISSING")

#         final_chunks = self._apply_mmr(
#             candidates,
#             query_embedding,
#             top_k
#         )

#         print("\n🏆 FINAL CHUNKS:")
#         for r in final_chunks:
#             print(
#                 f"{r['hybrid_score']:.4f} | "
#                 f"{r['source_file']} | "
#                 f"v{r.get('version', 1)}"
#             )

#         print("="*80)

#         return final_chunks


# # Global instance
# retrieval_service = RetrievalService()



#...............mmr code ended......................................................abs









import concurrent.futures
import re
import json
from rank_bm25 import BM25Okapi

from db import db
from embedding import embedding_service
from reranker import reranker_service

class RetrievalService:


    def __init__(self):
        self.WEIGHT_SEMANTIC = 0.7
        self.WEIGHT_BM25 = 0.3
        self.HYBRID_THRESHOLD = 0.005   # very low to send most chunks to reranker
        
        # BM25Okapi initialization
        self.bm25 = None
        self.bm25_docs = []
        self._load_bm25()

    # ──────────────────────────────────────────────
    # NORMALIZE BM25
    # ──────────────────────────────────────────────
    # def _normalize(self, scores):
    #     if not scores:
    #         return {}

    #     vals = list(scores.values())
    #     min_v, max_v = min(vals), max(vals)

    #     if max_v == min_v:
    #         return {k: 1.0 for k in scores}

    #     return {
    #         k: (v - min_v) / (max_v - min_v)
    #         for k, v in scores.items()
    #     }

    # ──────────────────────────────────────────────
    # LOAD BM25 CORPUS
    # ──────────────────────────────────────────────
    def _load_bm25(self):
        """Load all active chunks into BM25Okapi corpus"""
        try:
            if not db.ensure_connected():
                print("❌ Cannot load BM25: DB connection failed")
                return
                
            docs = db.get_all_active_chunks()
            
            if not docs:
                print("⚠️ No active chunks found for BM25")
                self.bm25 = None
                self.bm25_docs = []
                return
            
            self.bm25_docs = docs
            
            # ✅ IMPROVED: Tokenize content using regex for better word extraction
            tokenized = [
                re.findall(r"\w+", d["content"].lower())
                for d in docs
            ]
            
            # Debug: Show tokenization sample
            if tokenized:
                sample_tokens = tokenized[0][:10]  # First 10 tokens from first document
                print(f"🔧 Sample tokens: {sample_tokens}{'...' if len(tokenized[0]) > 10 else ''}")
            
            self.bm25 = BM25Okapi(tokenized)
            
            print(f"✅ BM25Okapi loaded with {len(docs)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to load BM25: {e}")
            self.bm25 = None
            self.bm25_docs = []

    # ──────────────────────────────────────────────
    # SEMANTIC SEARCH
    # ──────────────────────────────────────────────
    def _semantic_search(self, query_embedding, top_k):
        try:
            results = db.search_similar(query_embedding, top_k=top_k)
            print(f"🧠 Semantic search returned {len(results)} chunks")
            
            # 🔍 DETAILED SEMANTIC CHUNK DEBUG
            if results:
                print(f"🔧 Semantic score range: {results[0]['similarity_score']:.4f} to {results[-1]['similarity_score']:.4f}")
                print("\n🧠 SEMANTIC CHUNKS RETRIEVED:")
                print("-" * 80)
                for i, r in enumerate(results):
                    content_preview = r['content'][:100] + "..." if len(r['content']) > 100 else r['content']
                    print(f"   Chunk {i+1}: ID={r['id']} | Score={r['similarity_score']:.4f} | v{r.get('version', 1)}")
                    print(f"   Content: {content_preview}")
                    print(f"   File: {r.get('source_file', 'N/A')} | Folder: {r.get('folder', 'N/A')}")
                    print("-" * 80)
            
            return results
        except Exception as e:
            print(f"❌ Semantic error: {e}")
            return []

    # ──────────────────────────────────────────────
    # BM25 SEARCH (USING BM25OKAPI)
    # ──────────────────────────────────────────────
    def _bm25_search(self, query, top_k):
        try:
            if not self.bm25 or not self.bm25_docs:
                print("⚠️ BM25 corpus not loaded")
                return []
            
            # ✅ IMPROVED: Tokenize query using regex for better word extraction
            query_tokens = re.findall(r"\w+", query.lower())
            print(f"🔧 Query tokens: {query_tokens[:10]}{'...' if len(query_tokens) > 10 else ''}")  # Show first 10 tokens
            
            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)
            
            # Rank documents by score
            ranked = sorted(
                zip(self.bm25_docs, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            # Format results
            results = []
            for doc, score in ranked:
                if score > 0:  # Only include documents with positive scores
                    row = dict(doc)
                    row["bm25_score"] = float(score)
                    results.append(row)
            
            print(f"📝 BM25Okapi returned {len(results)} chunks" + (f" (top score: {results[0]['bm25_score']:.4f})" if results else ""))
            
            # 🔍 DETAILED BM25 CHUNK DEBUG
            if results:
                print(f"🔧 BM25 score range: {results[0]['bm25_score']:.4f} to {results[-1]['bm25_score']:.4f}")
                print("\n📝 BM25 CHUNKS RETRIEVED:")
                print("-" * 80)
                for i, r in enumerate(results):
                    content_preview = r['content'][:100] + "..." if len(r['content']) > 100 else r['content']
                    print(f"   Chunk {i+1}: ID={r['id']} | Score={r['bm25_score']:.4f} | v{r.get('version', 1)}")
                    print(f"   Content: {content_preview}")
                    print(f"   File: {r.get('source_file', 'N/A')} | Folder: {r.get('folder', 'N/A')}")
                    print("-" * 80)
            
            return results
            
        except Exception as e:
            print(f"❌ BM25Okapi error: {e}")
            return []

    # ──────────────────────────────────────────────
    # HYBRID FUSION
    # # ──────────────────────────────────────────────
    # def _fusion(self, semantic_results, bm25_results):

    #     semantic_scores = {
    #         r["id"]: float(r["similarity_score"])
    #         for r in semantic_results
    #     }

    #     bm25_scores = {
    #         r["id"]: float(r.get("bm25_score", 0.0))
    #         for r in bm25_results
    #     }

    #     bm25_norm = self._normalize(bm25_scores)

    #     all_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

    #     id_to_chunk = {}
    #     for r in semantic_results + bm25_results:
    #         if r["id"] not in id_to_chunk:
    #             id_to_chunk[r["id"]] = dict(r)

    #     fused = []

    #     for doc_id in all_ids:
    #         chunk = id_to_chunk[doc_id]

    #         s = semantic_scores.get(doc_id, 0.0)
    #         b = bm25_norm.get(doc_id, 0.0)

    #         chunk["semantic_score"] = s
    #         chunk["bm25_score"] = b

    #         chunk["hybrid_score"] = (
    #             self.WEIGHT_SEMANTIC * s +
    #             self.WEIGHT_BM25 * b
    #         )

    #         fused.append(chunk)

    #     return sorted(
    #         fused,
    #         key=lambda x: x["hybrid_score"],
    #         reverse=True
    #     )


    def _fusion(self, semantic_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion (RRF)
        Uses rank positions, not raw scores
        """

        # Merge chunk metadata
        id_to_chunk = {}

        for r in semantic_results + bm25_results:
            if r["id"] not in id_to_chunk:
                id_to_chunk[r["id"]] = dict(r)

        rrf_scores = {}

        # Semantic ranks
        for rank, row in enumerate(semantic_results, start=1):
            doc_id = row["id"]

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                1 / (k + rank)
            )

            id_to_chunk[doc_id]["semantic_rank"] = rank
            id_to_chunk[doc_id]["semantic_score"] = float(
                row.get("similarity_score", 0)
            )

        # BM25 ranks
        for rank, row in enumerate(bm25_results, start=1):
            doc_id = row["id"]

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                1 / (k + rank)
            )

            id_to_chunk[doc_id]["bm25_rank"] = rank
            id_to_chunk[doc_id]["bm25_score"] = float(
                row.get("bm25_score", 0)
            )

        fused = []

        for doc_id, score in rrf_scores.items():
            chunk = id_to_chunk[doc_id]

            chunk["hybrid_score"] = score   # keep same key for compatibility

            fused.append(chunk)

        return sorted(
            fused,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
    # ──────────────────────────────────────────────

    # MAIN SEARCH (SIMPLIFIED)
    # ──────────────────────────────────────────────
    def search(self, query, top_k=10, source_file=None):

        if not db.ensure_connected():
            print("❌ DB connection failed")
            return []

        print("\n" + "="*80)
        print("🔎 RRF HYBRID SEARCH (SIMPLIFIED)")
        print("="*80)
        print(f"Query: {query}")
        if source_file:
            print(f"🎯 File Filter: {source_file}")

        # STEP 1: Embedding
        query_embedding = embedding_service.embed_text(query)
        if query_embedding is None:
            return []

        fetch_k = top_k * 3

        # STEP 2: Parallel search
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            sem_future = executor.submit(
                self._semantic_search,
                query_embedding,
                fetch_k
            )
            bm25_future = executor.submit(
                self._bm25_search,
                query,
                fetch_k
            )

            semantic_results = sem_future.result()
            bm25_results = bm25_future.result()

        # 📊 PARALLEL SEARCH SUMMARY
        print(f"\n📊 PARALLEL SEARCH RESULTS:")
        print(f"   🧠 Semantic: {len(semantic_results)} chunks")
        print(f"   📝 BM25:     {len(bm25_results)} chunks")
        print(f"   🎯 Fetch-k:  {fetch_k} (requested per method)")

        # 🔍 STAGE 1: SEMANTIC RESULTS JSON
        print("\n" + "="*80)
        print("📊 STAGE 1: SEMANTIC SEARCH RESULTS (JSON)")
        print("="*80)
        semantic_json = {
            "stage": "1_semantic_search",
            "query": query,
            "total_chunks": len(semantic_results),
            "chunks": []
        }
        for i, r in enumerate(semantic_results):
            semantic_json["chunks"].append({
                "rank": i + 1,
                "chunk_id": r.get('id'),
                "semantic_score": round(r.get('similarity_score', 0), 4),
                "version": r.get('version', 1),
                "source_file": r.get('source_file', 'N/A'),
                "folder": r.get('folder', 'N/A'),
                "content_preview": r.get('content', '')[:150] + "..." if len(r.get('content', '')) > 150 else r.get('content', '')
            })
        print(json.dumps(semantic_json, indent=2, ensure_ascii=False))

        # 🔍 STAGE 2: BM25 RESULTS JSON
        print("\n" + "="*80)
        print("📊 STAGE 2: BM25 SEARCH RESULTS (JSON)")
        print("="*80)
        bm25_json = {
            "stage": "2_bm25_search",
            "query": query,
            "total_chunks": len(bm25_results),
            "chunks": []
        }
        for i, r in enumerate(bm25_results):
            bm25_json["chunks"].append({
                "rank": i + 1,
                "chunk_id": r.get('id'),
                "bm25_score": round(r.get('bm25_score', 0), 4),
                "version": r.get('version', 1),
                "source_file": r.get('source_file', 'N/A'),
                "folder": r.get('folder', 'N/A'),
                "content_preview": r.get('content', '')[:150] + "..." if len(r.get('content', '')) > 150 else r.get('content', '')
            })
        print(json.dumps(bm25_json, indent=2, ensure_ascii=False))

        # ✅ IMPROVEMENT 1: File-level filtering 
        if source_file:
            print(f"🔍 Filtering by file: {source_file}")
            semantic_results = [r for r in semantic_results if r["source_file"] == source_file]
            bm25_results = [r for r in bm25_results if r["source_file"] == source_file]
            print(f"📊 After filtering - Semantic: {len(semantic_results)}, BM25: {len(bm25_results)}")

        if not semantic_results and not bm25_results:
            print("❌ No results")
            return []

        # STEP 3: Hybrid fusion
        fused_results = self._fusion(
            semantic_results,
            bm25_results
        )

        # 🔍 STAGE 3: RRF HYBRID FUSION RESULTS JSON
        print("\n" + "="*80)
        print("📊 STAGE 3: RRF HYBRID FUSION RESULTS (JSON)")
        print("="*80)
        rrf_json = {
            "stage": "3_rrf_hybrid_fusion",
            "query": query,
            "total_chunks": len(fused_results),
            "chunks": []
        }
        for i, r in enumerate(fused_results):
            rrf_json["chunks"].append({
                "rank": i + 1,
                "chunk_id": r.get('id'),
                "semantic_score": round(r.get('semantic_score', 0), 4),
                "bm25_score": round(r.get('bm25_score', 0), 4),
                "rrf_hybrid_score": round(r.get('hybrid_score', 0), 4),
                "version": r.get('version', 1),
                "source_file": r.get('source_file', 'N/A'),
                "folder": r.get('folder', 'N/A'),
                "content_preview": r.get('content', '')[:150] + "..." if len(r.get('content', '')) > 150 else r.get('content', '')
            })
        print(json.dumps(rrf_json, indent=2, ensure_ascii=False))

        # ──────────────────────────────────────────────
        # 🔥 VERSION-AWARE BOOSTING
        # ──────────────────────────────────────────────
        latest_versions = {}

        for r in fused_results:
            file = r["source_file"]
            version = r.get("version", 1)
            latest_versions[file] = max(
                latest_versions.get(file, 0),
                version
            )

        for r in fused_results:
            file = r["source_file"]
            version = r.get("version", 1)
            latest = latest_versions[file]

            if version == latest:
                r["hybrid_score"] *= 1.15   # boost latest
            else:
                r["hybrid_score"] *= 0.85   # fallback allowed

        # ✅ FIX: Re-sort after boosting
        fused_results = sorted(
            fused_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        # ──────────────────────────────────────────────
        # STEP 4: Threshold filter
        # ──────────────────────────────────────────────
        filtered = [
            r for r in fused_results
            if r["hybrid_score"] >= self.HYBRID_THRESHOLD
        ]

        if not filtered:
            filtered = fused_results  # Send ALL fused results if none meet threshold

        # ──────────────────────────────────────────────
        # STEP 5: FINAL TOP-K SELECTION
        # ──────────────────────────────────────────────
        # final_chunks = sorted(
        #     filtered,
        #     key=lambda x: x["hybrid_score"],
        #     reverse=True
        # )[:top_k]
        # STEP 5: TAKE CANDIDATE POOL
        # STEP 5: candidate pool for reranker

        candidates = sorted(
            filtered,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )  # Send ALL RRF fusion results to reranker


        # STEP 6: apply reranker only if enough chunks

        if len(candidates) <= top_k:
            final_chunks = candidates

        else:
            final_chunks = reranker_service.rerank(
                query=query,
                chunks=candidates,
                top_n=top_k
            )

        # 🔍 STAGE 4: FINAL RERANKED RESULTS JSON
        print("\n" + "="*80)
        print("📊 STAGE 4: FINAL RERANKED RESULTS (JSON)")
        print("="*80)
        final_json = {
            "stage": "4_final_reranked_results",
            "query": query,
            "reranker_used": len(candidates) > top_k,
            "total_chunks": len(final_chunks),
            "chunks": []
        }
        for i, r in enumerate(final_chunks):
            final_json["chunks"].append({
                "final_rank": i + 1,
                "chunk_id": r.get('id'),
                "semantic_score": round(r.get('semantic_score', 0), 4),
                "bm25_score": round(r.get('bm25_score', 0), 4),
                "rrf_hybrid_score": round(r.get('hybrid_score', 0), 4),
                "reranker_applied": len(candidates) > top_k,
                "version": r.get('version', 1),
                "source_file": r.get('source_file', 'N/A'),
                "folder": r.get('folder', 'N/A'),
                "content_preview": r.get('content', '')[:150] + "..." if len(r.get('content', '')) > 150 else r.get('content', '')
            })
        print(json.dumps(final_json, indent=2, ensure_ascii=False))

        print("\n🏆 FINAL CHUNKS:")
        for r in final_chunks:
            print(
                f"{r['hybrid_score']:.4f} | "
                f"{r['source_file']} | "
                f"v{r.get('version', 1)}"
            )

        print("="*80)

        return final_chunks

    # ──────────────────────────────────────────────
    # RELOAD BM25 (CALL AFTER INGESTION/UPDATES)
    # ──────────────────────────────────────────────
    def reload_bm25(self):
        """Reload BM25 corpus after document changes"""
        print("🔄 Reloading BM25 corpus...")
        self._load_bm25()


# Global instance
retrieval_service = RetrievalService()

