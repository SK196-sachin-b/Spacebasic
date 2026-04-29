"""
Unified Cleanup Service

Handles both document and chat cleanup in a single service:
- Document chunks: Removes deactivated chunks after 15 days
- Chat history: Removes chat sessions and messages after 30 days

Uses dynamic scheduling to run cleanup exactly when needed.
"""

import threading
import time
import sys
import os
from datetime import datetime, timedelta

# Add scripts folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from db import db


class UnifiedCleanupScheduler:
    
    def __init__(self, document_retention_days=15, chat_retention_days=30):
        self.document_retention_days = document_retention_days
        self.chat_retention_days = chat_retention_days
        self.running = False
        self.thread = None
    
    # ──────────────────────────────────────────────
    # DOCUMENT CLEANUP FUNCTIONS
    # ──────────────────────────────────────────────
    
    def get_next_document_cleanup_time(self):
        """
        Get earliest document chunk that will expire (15 days after deactivation)
        """
        if not db.ensure_connected():
            return None
        
        try:
            db.cursor.execute("""
                SELECT MIN(updated_at) + INTERVAL '%s days' as next_cleanup
                FROM documents
                WHERE is_active = false
                AND updated_at IS NOT NULL;
            """, (self.document_retention_days,))
            
            result = db.cursor.fetchone()
            return result["next_cleanup"] if result and result["next_cleanup"] else None
            
        except Exception as e:
            print(f"❌ Error getting next document cleanup time: {e}")
            return None
    
    def cleanup_expired_documents(self):
        """
        Delete document chunks inactive for > retention period
        """
        if not db.ensure_connected():
            return 0
        
        try:
            db.cursor.execute("""
                DELETE FROM documents
                WHERE is_active = false
                AND updated_at < NOW() - INTERVAL '%s days';
            """, (self.document_retention_days,))
            
            deleted = db.cursor.rowcount
            db.connection.commit()
            
            if deleted > 0:
                print(f"🧹 Deleted {deleted} expired document chunks")
            
            return deleted
            
        except Exception as e:
            print(f"❌ Document cleanup error: {e}")
            return 0
    
    # ──────────────────────────────────────────────
    # CHAT CLEANUP FUNCTIONS
    # ──────────────────────────────────────────────
    
    def get_next_chat_cleanup_time(self):
        """
        Find the next time when chat cleanup should run
        """
        if not db.ensure_connected():
            return None
        
        try:
            # Check for sessions that are already old enough to delete
            db.cursor.execute("""
                SELECT MIN(created_at) + INTERVAL '%s days' as next_cleanup
                FROM chat_sessions
                WHERE created_at < NOW() - INTERVAL '%s days';
            """, (self.chat_retention_days, self.chat_retention_days))
            
            result = db.cursor.fetchone()
            
            if result and result["next_cleanup"]:
                return result["next_cleanup"]
            
            # If no old sessions, check when the oldest current session will expire
            db.cursor.execute("""
                SELECT MIN(created_at) + INTERVAL '%s days' as next_cleanup
                FROM chat_sessions;
            """, (self.chat_retention_days,))
            
            result = db.cursor.fetchone()
            return result["next_cleanup"] if result and result["next_cleanup"] else None
            
        except Exception as e:
            print(f"❌ Error getting next chat cleanup time: {e}")
            return None
    
    def cleanup_expired_chats(self):
        """
        Delete chat sessions and messages older than retention period
        """
        if not db.ensure_connected():
            return {"sessions": 0, "messages": 0}
        
        try:
            # First delete old messages (due to foreign key constraint)
            db.cursor.execute("""
                DELETE FROM chat_messages 
                WHERE session_id IN (
                    SELECT session_id 
                    FROM chat_sessions 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                );
            """, (self.chat_retention_days,))
            
            deleted_messages = db.cursor.rowcount
            
            # Then delete old sessions
            db.cursor.execute("""
                DELETE FROM chat_sessions 
                WHERE created_at < NOW() - INTERVAL '%s days';
            """, (self.chat_retention_days,))
            
            deleted_sessions = db.cursor.rowcount
            db.connection.commit()
            
            if deleted_sessions > 0 or deleted_messages > 0:
                print(f"🧹 Cleaned up {deleted_sessions} chat sessions and {deleted_messages} messages")
            
            return {"sessions": deleted_sessions, "messages": deleted_messages}
            
        except Exception as e:
            print(f"❌ Chat cleanup error: {e}")
            return {"sessions": 0, "messages": 0}
    
    # ──────────────────────────────────────────────
    # UNIFIED SCHEDULER
    # ──────────────────────────────────────────────
    
    def get_next_cleanup_time(self):
        """
        Get the earliest cleanup time (either document or chat)
        """
        doc_time = self.get_next_document_cleanup_time()
        chat_time = self.get_next_chat_cleanup_time()
        
        # Return the earliest time, or None if both are None
        if doc_time and chat_time:
            return min(doc_time, chat_time)
        elif doc_time:
            return doc_time
        elif chat_time:
            return chat_time
        else:
            return None
    
    def run_cleanup_cycle(self):
        """
        Run both document and chat cleanup
        """
        print("🔥 Running unified cleanup cycle...")
        
        # Run document cleanup
        doc_deleted = self.cleanup_expired_documents()
        
        # Run chat cleanup
        chat_result = self.cleanup_expired_chats()
        
        # Summary
        total_cleaned = doc_deleted + chat_result["sessions"] + chat_result["messages"]
        
        if total_cleaned > 0:
            print(f"✅ Cleanup completed: {doc_deleted} documents, {chat_result['sessions']} chat sessions, {chat_result['messages']} chat messages")
        else:
            print("ℹ️ No expired data found to clean up")
        
        return {
            "documents": doc_deleted,
            "chat_sessions": chat_result["sessions"],
            "chat_messages": chat_result["messages"]
        }
    
    def schedule_cleanup(self):
        """
        Continuously checks next cleanup time and sleeps until then
        """
        print(f"🧹 Unified cleanup scheduler started")
        print(f"📋 Document retention: {self.document_retention_days} days")
        print(f"💬 Chat retention: {self.chat_retention_days} days")
        
        while self.running:
            try:
                next_time = self.get_next_cleanup_time()
                
                if not next_time:
                    print("⏳ No data to clean up. Checking again in 6 hours...")
                    time.sleep(21600)  # 6 hours
                    continue
                
                now = datetime.now()
                
                # Handle timezone-aware datetime
                if next_time.tzinfo is not None:
                    import pytz
                    now = now.replace(tzinfo=pytz.UTC)
                
                wait_seconds = (next_time - now).total_seconds()
                
                if wait_seconds <= 0:
                    self.run_cleanup_cycle()
                    # Wait a bit before checking again
                    time.sleep(3600)  # 1 hour
                    continue
                
                # Limit wait time to max 24 hours for safety
                wait_seconds = min(wait_seconds, 86400)  # 24 hours max
                
                print(f"⏱ Next cleanup scheduled at: {next_time}")
                print(f"😴 Sleeping for {int(wait_seconds)} seconds ({wait_seconds/3600:.1f} hours)")
                
                time.sleep(wait_seconds)
                
            except Exception as e:
                print(f"❌ Unified cleanup scheduler error: {e}")
                time.sleep(3600)  # Wait 1 hour on error
    
    def start_scheduler(self):
        """
        Start the unified cleanup scheduler in a background thread
        """
        if self.running:
            print("⚠️ Unified cleanup scheduler already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.schedule_cleanup, daemon=True)
        self.thread.start()
        print("✅ Unified cleanup scheduler started in background")
        return True
    
    def stop_scheduler(self):
        """
        Stop the unified cleanup scheduler
        """
        if not self.running:
            print("⚠️ Unified cleanup scheduler not running")
            return False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✅ Unified cleanup scheduler stopped")
        return True
    
    def force_cleanup(self):
        """
        Force immediate cleanup (for manual triggers)
        """
        print("🔥 Force cleanup triggered...")
        result = self.run_cleanup_cycle()
        return result
    
    def get_cleanup_stats(self):
        """
        Get statistics about both document and chat cleanup
        """
        if not db.ensure_connected():
            return {}
        
        try:
            # Document stats
            db.cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(CASE WHEN is_active = false THEN 1 END) as inactive_documents,
                    COUNT(CASE WHEN is_active = false AND updated_at < NOW() - INTERVAL '%s days' THEN 1 END) as expired_documents
                FROM documents;
            """, (self.document_retention_days,))
            
            doc_stats = db.cursor.fetchone()
            
            # Chat stats
            db.cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN created_at < NOW() - INTERVAL '%s days' THEN 1 END) as expired_sessions,
                    MIN(created_at) as oldest_session,
                    MAX(created_at) as newest_session
                FROM chat_sessions;
            """, (self.chat_retention_days,))
            
            chat_stats = db.cursor.fetchone()
            
            db.cursor.execute("SELECT COUNT(*) as total_messages FROM chat_messages;")
            message_stats = db.cursor.fetchone()
            
            next_cleanup = self.get_next_cleanup_time()
            
            return {
                "documents": dict(doc_stats) if doc_stats else {},
                "chat_sessions": dict(chat_stats) if chat_stats else {},
                "chat_messages": dict(message_stats) if message_stats else {},
                "next_cleanup": next_cleanup,
                "document_retention_days": self.document_retention_days,
                "chat_retention_days": self.chat_retention_days,
                "scheduler_running": self.running
            }
            
        except Exception as e:
            print(f"❌ Error getting cleanup stats: {e}")
            return {}


# ──────────────────────────────────────────────
# GLOBAL INSTANCE
# ──────────────────────────────────────────────
unified_cleanup_scheduler = UnifiedCleanupScheduler(
    document_retention_days=15,
    chat_retention_days=30
)


# ──────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────

def start_unified_cleanup():
    """Start the unified cleanup scheduler"""
    return unified_cleanup_scheduler.start_scheduler()

def stop_unified_cleanup():
    """Stop the unified cleanup scheduler"""
    return unified_cleanup_scheduler.stop_scheduler()

def force_unified_cleanup():
    """Force immediate unified cleanup"""
    return unified_cleanup_scheduler.force_cleanup()

def get_unified_cleanup_stats():
    """Get unified cleanup statistics"""
    return unified_cleanup_scheduler.get_cleanup_stats()


# ──────────────────────────────────────────────
# LEGACY FUNCTIONS (for backward compatibility)
# ──────────────────────────────────────────────

def schedule_cleanup():
    """Legacy function - starts unified scheduler"""
    unified_cleanup_scheduler.schedule_cleanup()

def get_next_deletion_time():
    """Legacy function - gets next document cleanup time"""
    return unified_cleanup_scheduler.get_next_document_cleanup_time()

def delete_expired_chunks():
    """Legacy function - cleans up expired documents"""
    return unified_cleanup_scheduler.cleanup_expired_documents()


# ──────────────────────────────────────────────
# AUTO-START (OPTIONAL)
# ──────────────────────────────────────────────

def auto_start_unified_cleanup():
    """
    Auto-start unified cleanup when module is imported
    Call this from your main application
    """
    try:
        start_unified_cleanup()
        print("🧹 Unified cleanup auto-started successfully")
    except Exception as e:
        print(f"❌ Failed to auto-start unified cleanup: {e}")


if __name__ == "__main__":
    # Test the unified scheduler
    print("🧪 Testing Unified Cleanup Scheduler...")
    
    # Get stats
    stats = get_unified_cleanup_stats()
    print(f"📊 Current stats: {stats}")
    
    # Start scheduler
    start_unified_cleanup()
    
    # Keep running for testing
    try:
        while True:
            time.sleep(60)  # Check every minute
            stats = get_unified_cleanup_stats()
            print(f"📊 Documents: {stats['documents'].get('total_documents', 0)}, "
                  f"Chat Sessions: {stats['chat_sessions'].get('total_sessions', 0)}, "
                  f"Next cleanup: {stats.get('next_cleanup', 'N/A')}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping unified cleanup scheduler...")
        stop_unified_cleanup()