
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.cursor = None

    # ──────────────────────────────────────────────
    # CONNECT
    # ──────────────────────────────────────────────
    def connect(self):
        try:
            if self.connection and not self.connection.closed:
                return True

            self.connection = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT")
            )

            self.cursor = self.connection.cursor(
                cursor_factory=RealDictCursor
            )

            print("✅ Connected to PostgreSQL database")
            return True

        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    # ──────────────────────────────────────────────
    # ENSURE CONNECTION (SAFE)
    # ──────────────────────────────────────────────
    def ensure_connected(self):
        try:
            if self.connection and not self.connection.closed:
                with self.connection.cursor() as cur:
                    cur.execute("SELECT 1")
                return True

            print("⚠️ Reconnecting to DB...")
            return self.connect()

        except Exception:
            print("⚠️ Connection lost → reconnecting...")
            return self.connect()

    # ──────────────────────────────────────────────
    # CREATE TABLE
    # ──────────────────────────────────────────────
    def create_table(self):
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Main documents table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024),
                    content_tsv TSVECTOR,
                    source_file VARCHAR(255),
                    folder VARCHAR(50),
                    page_number INTEGER,
                    version INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index on content_tsv for fast text search
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_content_tsv_idx 
                ON documents USING GIN(content_tsv);
            """)

            # Chat sessions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chat messages table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    role VARCHAR(20),
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                );
            """)

            self.connection.commit()
            print("✅ Table ready")
            return True

        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            self.connection.rollback()
            return False

    # ──────────────────────────────────────────────
    # INSERT DOCUMENT
    # ──────────────────────────────────────────────
    def insert_document(self, content, embedding, source_file, folder=None, page_number=None, version=1):
        try:
            self.cursor.execute("""
                INSERT INTO documents (
                    content,
                    embedding,
                    content_tsv,
                    source_file,
                    folder,
                    page_number,
                    version,
                    is_active
                )
                VALUES (%s, %s, to_tsvector('english', %s), %s, %s, %s, %s, true)
                RETURNING id;
            """, (content, embedding, content, source_file, folder, page_number, version))

            doc_id = self.cursor.fetchone()["id"]
            self.connection.commit()
            print(f"✅ Inserted document with ID {doc_id} (content_tsv generated)")
            return doc_id

        except Exception as e:
            print(f"❌ Insert failed: {e}")
            self.connection.rollback()
            return None

    # ──────────────────────────────────────────────
    # SEMANTIC SEARCH (pgvector)
    # ──────────────────────────────────────────────
    def search_similar(self, query_embedding, top_k=15):
        try:
            print(f"🔧 DEBUG: search_similar called with top_k={top_k}")
            print(f"🔧 DEBUG: query_embedding type: {type(query_embedding)}")
            print(f"🔧 DEBUG: query_embedding length: {len(query_embedding) if query_embedding else 'None'}")
            
            self.cursor.execute("SET ivfflat.probes = 10;")
            query_vec = list(query_embedding)
            
            print(f"🔧 DEBUG: query_vec converted to list, length: {len(query_vec)}")

            self.cursor.execute("""
                SELECT id, content, embedding, source_file, folder, page_number, version, is_active,
                       1 - (embedding <=> %s::vector) AS similarity_score
                FROM documents
                WHERE is_active = true
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vec, query_vec, top_k))

            results = self.cursor.fetchall()
            print(f"🔧 DEBUG: Raw DB results count: {len(results)}")
            
            if results:
                print(f"🔧 DEBUG: First result similarity_score: {results[0].get('similarity_score', 'N/A')}")
                print(f"🔧 DEBUG: First result id: {results[0].get('id', 'N/A')}")
                print(f"🔧 DEBUG: First result has embedding: {bool(results[0].get('embedding'))}")
            
            converted_results = [dict(r) for r in results]
            print(f"🔧 DEBUG: Converted results count: {len(converted_results)}")
            
            return converted_results

        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ──────────────────────────────────────────────
    # BM25 SEARCH (THREAD SAFE)
    # ──────────────────────────────────────────────
    def search_bm25(self, query, top_k=15):
        """
        IMPORTANT:
        Uses NEW connection → avoids threading issues
        """

        try:
            # 🔥 NEW CONNECTION (fix)
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT")
            )

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            print(f"🔧 DEBUG: BM25 query='{query}'")

            # Count check
            cursor.execute("SELECT COUNT(*) as count FROM documents;")
            result = cursor.fetchone()

            if not result:
                print("❌ BM25: No result from count query")
                cursor.close()
                conn.close()
                return []

            doc_count = result.get("count", 0)

            if doc_count == 0:
                print("⚠️ BM25: No documents")
                cursor.close()
                conn.close()
                return []

            # BM25 query
            cursor.execute("""
                SELECT
                    id,
                    content,
                    source_file,
                    folder,
                    version,
                    is_active,
                    ts_rank_cd(
                        content_tsv,
                        websearch_to_tsquery('english', %s)
                    ) AS bm25_score
                FROM documents
                WHERE is_active = true
                AND content_tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY bm25_score DESC
                LIMIT %s;
            """, (query, query, top_k))

            results = cursor.fetchall()

            print(f"📝 BM25 returned {len(results)} results")

            cursor.close()
            conn.close()

            return [dict(r) for r in results]

        except Exception as e:
            print(f"❌ BM25 failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_active_chunks(self):
        """
        Fetch all active chunks for BM25Okapi corpus building
        """
        try:
            self.cursor.execute("""
                SELECT id, content, source_file, folder,
                       page_number, version, is_active
                FROM documents
                WHERE is_active = true
                ORDER BY id;
            """)
            results = self.cursor.fetchall()
            print(f"📊 Fetched {len(results)} active chunks for BM25 corpus")
            return [dict(r) for r in results]
        except Exception as e:
            print(f"❌ Failed to fetch active chunks: {e}")
            return []
    
    def get_next_version(self, source_file):
        try:
            query = """
            SELECT COALESCE(MAX(version), 0) + 1 AS version
            FROM documents
            WHERE source_file = %s;
            """

            self.cursor.execute(query, (source_file,))
            result = self.cursor.fetchone()

            version = result["version"]

            print(f"🆕 Next version for {source_file}: {version}")
            return version

        except Exception as e:
            print(f"❌ Version fetch failed: {e}")
            return 1

    def create_index_after_ingestion(self):
        try:
            print("🔄 Creating vector index...")

            # Drop old index (safe)
            self.cursor.execute(
                "DROP INDEX IF EXISTS documents_embedding_idx;"
            )
            self.connection.commit()

            # Count rows
            self.cursor.execute("SELECT COUNT(*) as count FROM documents;")
            result = self.cursor.fetchone()

            if not result or result["count"] == 0:
                print("⚠️ No data found, skipping index")
                return False

            count = result["count"]

            # Optimal lists = sqrt(N)
            lists = max(1, int(count ** 0.5))

            print(f"📊 Total chunks: {count} | IVFFlat lists: {lists}")

            self.cursor.execute(f"""
                CREATE INDEX documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)

            self.connection.commit()
            print("✅ Vector index created successfully")
            return True

        except Exception as e:
            print(f"❌ Index creation failed: {e}")
            self.connection.rollback()
            return False
    # ──────────────────────────────────────────────
    # CLEAR TABLE
    # ──────────────────────────────────────────────
    def clear_documents(self):
        try:
            self.cursor.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
            self.connection.commit()
            print("✅ Documents cleared")
            return True

        except Exception as e:
            print(f"❌ Clear failed: {e}")
            self.connection.rollback()
            return False

    # ──────────────────────────────────────────────
    # CLOSE
    # ──────────────────────────────────────────────
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("✅ DB connection closed")

    # ──────────────────────────────────────────────
    # CHAT SESSION MANAGEMENT
    # ──────────────────────────────────────────────
    
    def create_session(self):
        """Create a new chat session and return session_id"""
        import uuid
        
        try:
            session_id = str(uuid.uuid4())
            
            self.cursor.execute("""
                INSERT INTO chat_sessions (session_id)
                VALUES (%s);
            """, (session_id,))
            
            self.connection.commit()
            print(f"✅ Created new session: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"❌ Failed to create session: {e}")
            return None
    
    def store_message(self, session_id, role, message):
        """Store a message in the chat history"""
        try:
            self.cursor.execute("""
                INSERT INTO chat_messages (session_id, role, message)
                VALUES (%s, %s, %s);
            """, (session_id, role, message))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ Failed to store message: {e}")
            return False
    
    def get_chat_history(self, session_id, limit=6):
        """Get chat history for a session (last N messages)"""
        try:
            self.cursor.execute("""
                SELECT role, message, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT %s;
            """, (session_id, limit))
            
            results = self.cursor.fetchall()
            return [dict(r) for r in results]
            
        except Exception as e:
            print(f"❌ Failed to get chat history: {e}")
            return []
    
    def get_all_sessions(self):
        """Get all chat sessions"""
        try:
            self.cursor.execute("""
                SELECT s.session_id, s.created_at,
                       COUNT(m.id) as message_count,
                       MAX(m.created_at) as last_message
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                GROUP BY s.session_id, s.created_at
                ORDER BY last_message DESC NULLS LAST;
            """)
            
            results = self.cursor.fetchall()
            return [dict(r) for r in results]
            
        except Exception as e:
            print(f"❌ Failed to get sessions: {e}")
            return []

    def deactivate_chunks(self, ids):
        """
        Deactivate chunks by setting is_active = false
        Used for content updates - old chunks become inactive
        """
        try:
            self.cursor.execute("""
                UPDATE documents
                SET is_active = false,
                    updated_at = NOW()
                WHERE id = ANY(%s);
            """, (ids,))
            self.connection.commit()
            print(f"✅ Deactivated {len(ids)} chunks: {ids}")
            return True
        except Exception as e:
            print(f"❌ Deactivation failed: {e}")
            self.connection.rollback()
            return False

    # ──────────────────────────────────────────────
    # CHAT HISTORY CLEANUP FUNCTIONS
    # ──────────────────────────────────────────────
    
    def cleanup_old_chat_sessions(self, days_old=30):
        """
        Delete chat sessions and messages older than specified days
        """
        try:
            # First delete old messages (due to foreign key constraint)
            self.cursor.execute("""
                DELETE FROM chat_messages 
                WHERE session_id IN (
                    SELECT session_id 
                    FROM chat_sessions 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                );
            """, (days_old,))
            
            deleted_messages = self.cursor.rowcount
            
            # Then delete old sessions
            self.cursor.execute("""
                DELETE FROM chat_sessions 
                WHERE created_at < NOW() - INTERVAL '%s days';
            """, (days_old,))
            
            deleted_sessions = self.cursor.rowcount
            self.connection.commit()
            
            print(f"🧹 Cleaned up {deleted_sessions} sessions and {deleted_messages} messages older than {days_old} days")
            return {"sessions": deleted_sessions, "messages": deleted_messages}
            
        except Exception as e:
            print(f"❌ Chat cleanup failed: {e}")
            self.connection.rollback()
            return {"sessions": 0, "messages": 0}
    
    def get_next_chat_cleanup_time(self, days_old=30):
        """
        Find the next time when chat cleanup should run
        """
        try:
            self.cursor.execute("""
                SELECT MIN(created_at) + INTERVAL '%s days' as next_cleanup
                FROM chat_sessions
                WHERE created_at < NOW() - INTERVAL '%s days';
            """, (days_old, days_old))
            
            result = self.cursor.fetchone()
            
            if result and result["next_cleanup"]:
                return result["next_cleanup"]
            
            # If no old sessions, check when the oldest current session will expire
            self.cursor.execute("""
                SELECT MIN(created_at) + INTERVAL '%s days' as next_cleanup
                FROM chat_sessions;
            """, (days_old,))
            
            result = self.cursor.fetchone()
            return result["next_cleanup"] if result else None
            
        except Exception as e:
            print(f"❌ Error getting next cleanup time: {e}")
            return None
    
    def get_chat_cleanup_stats(self):
        """
        Get statistics about chat sessions for monitoring
        """
        try:
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN created_at < NOW() - INTERVAL '30 days' THEN 1 END) as old_sessions,
                    MIN(created_at) as oldest_session,
                    MAX(created_at) as newest_session
                FROM chat_sessions;
            """)
            
            session_stats = self.cursor.fetchone()
            
            self.cursor.execute("""
                SELECT COUNT(*) as total_messages
                FROM chat_messages;
            """)
            
            message_stats = self.cursor.fetchone()
            
            return {
                "sessions": dict(session_stats) if session_stats else {},
                "messages": dict(message_stats) if message_stats else {}
            }
            
        except Exception as e:
            print(f"❌ Error getting cleanup stats: {e}")
            return {"sessions": {}, "messages": {}}


# ✅ Global instance
db = DatabaseConnection()