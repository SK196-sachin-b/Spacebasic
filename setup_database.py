#!/usr/bin/env python3
"""
Database setup script for SpaceBasic RAG system
Creates all necessary tables and indexes
"""

import sys
import os

# Add scripts folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from db import db

def setup_database():
    """Create all database tables and indexes"""
    
    print("🚀 Setting up SpaceBasic RAG Database")
    print("=" * 50)
    
    if not db.connect():
        print("❌ Failed to connect to database")
        return False
    
    try:
        # Create documents table
        print("📄 Creating documents table...")
        if db.create_table():
            print("✅ Documents table created successfully")
        else:
            print("❌ Failed to create documents table")
            return False
        
        # Create chat sessions table
        print("💬 Creating chat sessions table...")
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create chat messages table
        print("📝 Creating chat messages table...")
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        print("🔍 Creating indexes...")
        
        # Index for chat messages by session
        db.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time 
            ON chat_messages(session_id, created_at);
        """)
        
        # Index for documents content_tsv (if column exists)
        try:
            db.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_content_tsv 
                ON documents USING GIN(content_tsv);
            """)
            print("✅ Content TSV index created")
        except Exception as e:
            print(f"⚠️ Content TSV index not created (column may not exist): {e}")
        
        # Commit all changes
        db.connection.commit()
        print("✅ All tables and indexes created successfully")
        
        # Show table info
        print("\n📊 Database Schema Summary:")
        print("-" * 30)
        
        # Count documents
        db.cursor.execute("SELECT COUNT(*) FROM documents WHERE is_active = true;")
        doc_count = db.cursor.fetchone()[0]
        print(f"📄 Active documents: {doc_count}")
        
        # Count sessions
        db.cursor.execute("SELECT COUNT(*) FROM chat_sessions;")
        session_count = db.cursor.fetchone()[0]
        print(f"💬 Chat sessions: {session_count}")
        
        # Count messages
        db.cursor.execute("SELECT COUNT(*) FROM chat_messages;")
        message_count = db.cursor.fetchone()[0]
        print(f"📝 Chat messages: {message_count}")
        
        print("\n✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        db.close()

if __name__ == "__main__":
    setup_database()