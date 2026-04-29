#!/usr/bin/env python3
"""
STEP 5: Update Service for Role-based Content Management
"""

from retrieval import retrieval_service
from embedding import embedding_service
from db import db

def create_content(user_role, file_name, new_content):
    """
    Create completely new content without checking for existing chunks
    """
    print(f"\n📝 CREATE NEW CONTENT")
    print(f"👤 Role: {user_role}")
    print(f"📄 File: {file_name}")
    print(f"📝 Content length: {len(new_content)} chars")
    print("="*50)
    
    # Check authorization
    if user_role != "staff":
        return "❌ Unauthorized: Only staff members can create content"
    
    if not db.connect():
        return "❌ Database connection failed"
    
    try:
        print("📝 Creating new content directly...")
        return _insert_new_content(file_name, new_content)
        
    except Exception as e:
        print(f"❌ Create failed: {e}")
        return f"❌ Create failed: {str(e)}"
    finally:
        db.close()


def preview_delete_by_content(user_role, file_name, search_content):
    """
    Preview chunks for deletion based on content similarity
    Uses retrieval to find similar chunks, then user selects which to delete
    """
    print(f"\n🔍 PREVIEW DELETE BY CONTENT")
    print(f"👤 Role: {user_role}")
    print(f"📄 File: {file_name}")
    print(f"🔍 Search content length: {len(search_content)} chars")
    print("="*50)
    
    # Check authorization
    if user_role != "staff":
        return {
            "status": "error",
            "message": "❌ Unauthorized: Only staff members can delete content"
        }
    
    if not db.connect():
        return {
            "status": "error", 
            "message": "❌ Database connection failed"
        }
    
    try:
        print("\n🔍 Step 1: Finding similar chunks...")
        
        # Use shorter query for better performance (same as update)
        search_query = search_content[:200] + "..." if len(search_content) > 200 else search_content
        print(f"🔍 Using search query (first 200 chars): {search_query}")
        
        # Use retrieval service to find similar chunks
        results = retrieval_service.search(search_query, top_k=10, source_file=file_name)
        
        # If no results with content search, try file-based search
        if not results:
            print("🔄 No results with content search, trying file-based search...")
            file_keywords = file_name.replace(".pdf", "").replace("_", " ").replace("-", " ")
            results = retrieval_service.search(file_keywords, top_k=10, source_file=file_name)
        
        if not results:
            print("📝 No similar chunks found")
            return {
                "status": "no_chunks",
                "message": f"No chunks similar to your content found in {file_name}",
                "chunks": [],
                "file_name": file_name,
                "search_content": search_content
            }
        
        # Format chunks for preview with similarity scores
        chunk_list = []
        for r in results:
            score = r.get("hybrid_score", r.get("similarity_score", 0))
            chunk_preview = {
                "id": r["id"],
                "score": round(score, 4),
                "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                "full_content": r["content"],
                "source_file": r["source_file"],
                "folder": r.get("folder", ""),
                "version": r.get("version", 1),
                "page_number": r.get("page_number")
            }
            chunk_list.append(chunk_preview)
        
        print(f"🎯 Found {len(chunk_list)} similar chunks for potential deletion")
        return {
            "status": "chunks_found",
            "message": f"Found {len(chunk_list)} similar chunks",
            "chunks": chunk_list,
            "file_name": file_name,
            "search_content": search_content
        }
            
    except Exception as e:
        print(f"❌ Preview delete failed: {e}")
        return {
            "status": "error",
            "message": f"❌ Preview delete failed: {str(e)}"
        }
    finally:
        db.close()


def preview_delete(user_role, file_name):
    """
    Preview what chunks can be deleted from a file
    """
    print(f"\n🗑️ PREVIEW DELETE REQUEST")
    print(f"👤 Role: {user_role}")
    print(f"📄 File: {file_name}")
    print("="*50)
    
    # Check authorization
    if user_role != "staff":
        return {
            "status": "error",
            "message": "❌ Unauthorized: Only staff members can delete content"
        }
    
    if not db.connect():
        return {
            "status": "error", 
            "message": "❌ Database connection failed"
        }
    
    try:
        print("\n🔍 Finding chunks to delete...")
        
        # Get all active chunks for this file
        db.cursor.execute("""
            SELECT id, content, source_file, folder, version, page_number
            FROM documents 
            WHERE source_file = %s AND is_active = true
            ORDER BY id;
        """, (file_name,))
        
        chunks = db.cursor.fetchall()
        
        if not chunks:
            print("📝 No active chunks found")
            return {
                "status": "no_chunks",
                "message": f"No active chunks found in {file_name}",
                "chunks": [],
                "file_name": file_name
            }
        
        # Format chunks for preview
        chunk_list = []
        for chunk in chunks:
            chunk_preview = {
                "id": chunk["id"],
                "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "full_content": chunk["content"],
                "source_file": chunk["source_file"],
                "folder": chunk.get("folder", ""),
                "version": chunk.get("version", 1),
                "page_number": chunk.get("page_number")
            }
            chunk_list.append(chunk_preview)
        
        print(f"🎯 Found {len(chunk_list)} chunks for deletion")
        return {
            "status": "chunks_found",
            "message": f"Found {len(chunk_list)} chunks that can be deleted",
            "chunks": chunk_list,
            "file_name": file_name
        }
            
    except Exception as e:
        print(f"❌ Preview delete failed: {e}")
        return {
            "status": "error",
            "message": f"❌ Preview delete failed: {str(e)}"
        }
    finally:
        db.close()


def confirm_delete(preview_data):
    """
    Execute the delete operation for selected chunks only
    """
    print(f"\n🗑️ CONFIRMING SELECTIVE DELETE")
    print(f"📄 File: {preview_data['file_name']}")
    
    if preview_data["status"] == "error":
        return preview_data["message"]
    
    if preview_data["status"] == "no_chunks":
        return "❌ No chunks found to delete"
    
    # Get selected chunks
    selected_chunks = preview_data.get("selected_chunks", [])
    
    if not selected_chunks:
        return "❌ No chunks selected for deletion"
    
    print(f"🎯 Selected chunks to delete: {len(selected_chunks)}")
    
    if not db.connect():
        return "❌ Database connection failed"
    
    try:
        # Deactivate only selected chunks
        chunk_ids = [chunk["id"] for chunk in selected_chunks]
        print(f"🔥 Deactivating selected chunks: {chunk_ids}")
        
        if not db.deactivate_chunks(chunk_ids):
            return "❌ Failed to deactivate selected chunks"
        
        total_chunks = len(preview_data["chunks"])
        deleted_count = len(chunk_ids)
        remaining_count = total_chunks - deleted_count
        
        # 🔥 Reload BM25 corpus after delete
        print("🔄 Reloading BM25 corpus...")
        from scripts.retrieval import retrieval_service
        retrieval_service.reload_bm25()
        
        return f"✅ Successfully deleted {deleted_count} selected chunks from {preview_data['file_name']}. Remaining active chunks: {remaining_count}. Deactivated chunk IDs: {chunk_ids}"
        
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        return f"❌ Delete failed: {str(e)}"
    finally:
        db.close()


def preview_update(user_role, file_name, new_content):
    """
    Preview what chunks will be affected by an update
    Returns information for user to review before confirming
    """
    print(f"\n🔍 PREVIEW UPDATE REQUEST")
    print(f"👤 Role: {user_role}")
    print(f"📄 File: {file_name}")
    print(f"📝 Content length: {len(new_content)} chars")
    print("="*50)
    
    # Check authorization
    if user_role != "staff":
        return {
            "status": "error",
            "message": "❌ Unauthorized: Only staff members can update content"
        }
    
    if not db.connect():
        return {
            "status": "error", 
            "message": "❌ Database connection failed"
        }
    
    try:
        print("\n🔍 Step 1: Finding relevant chunks...")
        
        # Use shorter query for better performance
        search_query = new_content[:200] + "..." if len(new_content) > 200 else new_content
        print(f"🔍 Using search query (first 200 chars): {search_query}")
        
        # Search for relevant chunks
        results = retrieval_service.search(search_query, top_k=5, source_file=file_name)
        
        # If no results with content search, try file-based search
        if not results:
            print("🔄 No results with content search, trying file-based search...")
            file_keywords = file_name.replace(".pdf", "").replace("_", " ").replace("-", " ")
            results = retrieval_service.search(file_keywords, top_k=5, source_file=file_name)
        
        if not results:
            print("📝 No similar content found")
            return {
                "status": "no_chunks",
                "message": f"No existing chunks found in {file_name}",
                "action": "insert_new",
                "chunks": [],
                "file_name": file_name,
                "new_content": new_content
            }
        
        # Filter by similarity threshold
        relevant_results = []
        low_score_results = []
        
        for r in results:
            score = r.get("hybrid_score", r.get("similarity_score", 0))
            chunk_preview = {
                "id": r["id"],
                "score": round(score, 4),
                "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                "full_content": r["content"],
                "source_file": r["source_file"],
                "folder": r.get("folder", ""),
                "version": r.get("version", 1)
            }
            
            if score >= 0.2:
                relevant_results.append(chunk_preview)
            else:
                low_score_results.append(chunk_preview)
        
        if relevant_results:
            print(f"🎯 Found {len(relevant_results)} relevant chunks for update")
            return {
                "status": "chunks_found",
                "message": f"Found {len(relevant_results)} chunks that will be updated",
                "action": "update_existing", 
                "chunks": relevant_results,
                "low_score_chunks": low_score_results,
                "file_name": file_name,
                "new_content": new_content
            }
        else:
            print(f"⚠️ Found chunks but all have low similarity scores")
            return {
                "status": "low_similarity",
                "message": f"Found {len(low_score_results)} chunks but similarity scores are low (< 0.2)",
                "action": "insert_new",
                "chunks": [],
                "low_score_chunks": low_score_results,
                "file_name": file_name,
                "new_content": new_content
            }
            
    except Exception as e:
        print(f"❌ Preview failed: {e}")
        return {
            "status": "error",
            "message": f"❌ Preview failed: {str(e)}"
        }
    finally:
        db.close()


def confirm_update(preview_data, force_update=False):
    """
    Execute the update after user confirmation
    """
    print(f"\n✅ CONFIRMING UPDATE")
    print(f"📄 File: {preview_data['file_name']}")
    print(f"🎯 Action: {preview_data['action']}")
    print("="*50)
    
    if preview_data["status"] == "error":
        return preview_data["message"]
    
    if not db.connect():
        return "❌ Database connection failed"
    
    try:
        if preview_data["action"] == "insert_new":
            print("📝 Inserting as new content...")
            return _insert_new_content(preview_data["file_name"], preview_data["new_content"])
            
        elif preview_data["action"] == "update_existing":
            print(f"🔄 Updating existing chunks...")
            
            # ✅ IMPROVEMENT: Use selected chunks if available
            chunks_to_update = preview_data.get("selected_chunks", preview_data.get("chunks", []))
            
            if not chunks_to_update:
                return "❌ No chunks selected for update"
            
            print(f"🎯 Selected chunks to update: {len(chunks_to_update)}")
            
            # Deactivate selected chunks only
            chunk_ids = [chunk["id"] for chunk in chunks_to_update]
            print(f"🔥 Deactivating selected chunks: {chunk_ids}")
            
            if not db.deactivate_chunks(chunk_ids):
                return "❌ Failed to deactivate selected chunks"
            
            # Create new chunk
            print("✨ Creating new chunk...")
            ref_chunk = chunks_to_update[0]  # Use first selected chunk as reference
            
            # Generate embedding
            embedding = embedding_service.embed_text(preview_data["new_content"])
            if embedding is None:
                return "❌ Failed to generate embedding"
            
            # Get next version
            new_version = db.get_next_version(preview_data["file_name"])
            
            # Insert new chunk
            doc_id = db.insert_document(
                content=preview_data["new_content"],
                embedding=list(embedding),
                source_file=ref_chunk["source_file"],
                folder=ref_chunk["folder"],
                page_number=None,
                version=new_version
            )
            
            if doc_id:
                return f"✅ Successfully updated content in {preview_data['file_name']}. Deactivated {len(chunk_ids)} selected chunks: {chunk_ids}, Created new chunk: {doc_id}"
            else:
                return "❌ Failed to insert new content"
        
        else:
            return f"❌ Unknown action: {preview_data['action']}"
            
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return f"❌ Update failed: {str(e)}"
    finally:
        db.close()


def update_content(user_role, file_name, new_content, force_update=False):
    """
    Update content with role-based access control
    
    Args:
        user_role: "student" or "staff"
        file_name: Name of the file to update (e.g., "policy.pdf")
        new_content: New content to add/update
    
    Returns:
        Status message
    """
    print(f"\n🔄 UPDATE REQUEST")
    print(f"👤 Role: {user_role}")
    print(f"📄 File: {file_name}")
    print(f"📝 Content length: {len(new_content)} chars")
    print("="*50)
    
    # Check authorization
    if user_role != "staff":
        print("❌ Access denied - only staff can update content")
        return "❌ Unauthorized: Only staff members can update content"
    
    if not db.connect():
        print("❌ Database connection failed")
        return "❌ Database connection failed"
    
    try:
        print("\n🔍 Step 1: Finding relevant chunks...")
        
        # ✅ IMPROVEMENT: Use shorter query for better BM25 performance
        # Extract key terms from long content for search
        search_query = new_content[:200] + "..." if len(new_content) > 200 else new_content
        print(f"🔍 Using search query (first 200 chars): {search_query}")
        
        # ✅ IMPROVEMENT: Use file-level filtering for better performance
        results = retrieval_service.search(search_query, top_k=5, source_file=file_name)
        
        # ✅ If no results with short query, try with file name keywords
        if not results:
            print("🔄 No results with content search, trying file-based search...")
            file_keywords = file_name.replace(".pdf", "").replace("_", " ").replace("-", " ")
            results = retrieval_service.search(file_keywords, top_k=5, source_file=file_name)
        
        if not results:
            print("📝 No similar content found - inserting as new")
            return _insert_new_content(file_name, new_content)
        
        
        # ✅ SAFETY CHECK: Add threshold to prevent wrong updates
        relevant_results = []
        for r in results:
            score = r.get("hybrid_score", r.get("similarity_score", 0))
            if score >= 0.2:  # ✅ Lowered threshold from 0.3 to 0.2 for more flexible updates
                relevant_results.append(r)
            else:
                print(f"⚠️ Low similarity score {score:.4f} for chunk {r['id']} - skipping")
        
        if not relevant_results:
            if force_update and results:
                print(f"🔧 Force update enabled - using all results regardless of score")
                relevant_results = results
            else:
                print(f"📝 No relevant content found (all scores < 0.2) - inserting as new")
                return _insert_new_content(file_name, new_content)
        
        print(f"🎯 Found {len(relevant_results)} relevant chunks in {file_name}")
        
        # Show what will be updated
        for i, r in enumerate(relevant_results):
            score = r.get("hybrid_score", r.get("similarity_score", 0))
            print(f"   Chunk {i+1}: ID={r['id']}, Score={score:.4f}")
            print(f"   Preview: {r['content'][:100]}...")
        
        print(f"\n🔥 Step 2: Deactivating old chunks...")
        
        # Deactivate old chunks
        ids = [r["id"] for r in relevant_results]
        if not db.deactivate_chunks(ids):
            return "❌ Failed to deactivate old chunks"
        
        print(f"\n✨ Step 3: Creating new chunk...")
        
        # Get metadata from the first result
        ref = relevant_results[0]
        
        # Generate embedding for new content
        print("🧠 Generating embedding...")
        embedding = embedding_service.embed_text(new_content)
        if embedding is None:
            return "❌ Failed to generate embedding"
        
        # Insert updated chunk with incremented version
        new_version = db.get_next_version(file_name)
        
        doc_id = db.insert_document(
            content=new_content,
            embedding=list(embedding),
            source_file=ref["source_file"],
            folder=ref["folder"],
            page_number=ref.get("page_number"),
            version=new_version
        )
        
        if doc_id:
            print(f"✅ Created new chunk with ID: {doc_id}")
            
            # 🔥 Reload BM25 corpus after update
            print("🔄 Reloading BM25 corpus...")
            from scripts.retrieval import retrieval_service
            retrieval_service.reload_bm25()
            
            return f"✅ Successfully updated content in {file_name}. Deactivated chunks: {ids}, Created new chunk: {doc_id}"
        else:
            return "❌ Failed to insert new content"
            
    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Update failed: {str(e)}"
    finally:
        db.close()

def _insert_new_content(file_name, new_content):
    """Insert completely new content"""
    try:
        print("🧠 Generating embedding for new content...")
        embedding = embedding_service.embed_text(new_content)
        if embedding is None:
            return "❌ Failed to generate embedding"
        
        # Determine folder based on file name
        folder = "manual_update"
        if "policy" in file_name.lower():
            folder = "policy"
        elif "faq" in file_name.lower():
            folder = "FAQ"
        elif "manual" in file_name.lower():
            folder = "user_manuals"
        
        doc_id = db.insert_document(
            content=new_content,
            embedding=list(embedding),
            source_file=file_name,
            folder=folder,
            page_number=None,
            version=1
        )
        
        if doc_id:
            print(f"✅ Inserted new content with ID: {doc_id}")
            
            # 🔥 Reload BM25 corpus after create
            print("🔄 Reloading BM25 corpus...")
            from scripts.retrieval import retrieval_service
            retrieval_service.reload_bm25()
            
            return f"✅ Successfully added new content to {file_name}. New chunk ID: {doc_id}"
        else:
            return "❌ Failed to insert new content"
            
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return f"❌ Insert failed: {str(e)}"

def get_user_role_info():
    """Get information about user roles"""
    return {
        "student": {
            "permissions": ["query"],
            "description": "Can ask questions and search documents"
        },
        "staff": {
            "permissions": ["query", "update"],
            "description": "Can ask questions, search documents, and update content"
        }
    }

# Test function
def test_update_service():
    """Test the update service"""
    print("🧪 Testing Update Service...")
    
    # Test unauthorized access
    result1 = update_content("student", "test.pdf", "Test content")
    print(f"Student access test: {result1}")
    
    # Test authorized access (would need valid content)
    # result2 = update_content("staff", "policy.pdf", "Updated policy content")
    # print(f"Staff access test: {result2}")

if __name__ == "__main__":
    test_update_service()