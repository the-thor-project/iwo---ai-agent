"""
Database Module - Handle conversation persistence
SQLite database for storing conversations and user data
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Database:
    """SQLite database handler for conversation storage"""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _create_tables(self):
        """Create necessary database tables"""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                preferences JSON
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                metadata JSON,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                metadata JSON,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user 
            ON messages(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user 
            ON sessions(user_id)
        """)
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    def create_user(self, user_id: str, preferences: dict = None) -> bool:
        """Create a new user"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO users (user_id, preferences)
                VALUES (?, ?)
            """, (user_id, json.dumps(preferences or {})))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def create_session(self, user_id: str, title: str = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: User identifier
            title: Optional session title
            
        Returns:
            Session ID
        """
        try:
            import uuid
            session_id = str(uuid.uuid4())
            
            # Ensure user exists
            self.create_user(user_id)
            
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, title)
                VALUES (?, ?, ?)
            """, (session_id, user_id, title or f"Session_{session_id[:8]}"))
            self.connection.commit()
            
            logger.info(f"Session created: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    def save_message(self, user_id: str, message: str, response: str, 
                    session_id: str = None, tokens_used: int = 0) -> str:
        """
        Save a user message and bot response
        
        Args:
            user_id: User identifier
            message: User message
            response: Bot response
            session_id: Session ID (create new if None)
            tokens_used: Number of tokens used
            
        Returns:
            Session ID
        """
        try:
            # Create session if needed
            if session_id is None:
                session_id = self.create_session(user_id)
            
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (session_id, user_id, user_message, bot_response, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, message, response, tokens_used))
            
            # Update session timestamp
            cursor.execute("""
                UPDATE sessions 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            
            self.connection.commit()
            return session_id
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            raise
    
    def get_messages(self, user_id: str, session_id: str = None, 
                    limit: int = 50) -> List[Dict]:
        """
        Retrieve conversation messages
        
        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Number of messages to retrieve
            
        Returns:
            List of messages with user and bot messages
        """
        try:
            cursor = self.connection.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM messages 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM messages 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
        except Exception as e:
            logger.error(f"Error retrieving messages: {str(e)}")
            return []
    
    def get_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get all sessions for a user"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving sessions: {str(e)}")
            return []
    
    def delete_messages(self, user_id: str, session_id: str = None) -> bool:
        """Delete messages for a user or session"""
        try:
            cursor = self.connection.cursor()
            
            if session_id:
                cursor.execute("""
                    DELETE FROM messages 
                    WHERE user_id = ? AND session_id = ?
                """, (user_id, session_id))
            else:
                cursor.execute("""
                    DELETE FROM messages 
                    WHERE user_id = ?
                """, (user_id,))
            
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting messages: {str(e)}")
            return False
    
    def get_statistics(self, user_id: str) -> Dict:
        """Get conversation statistics"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as total_messages FROM messages WHERE user_id = ?
            """, (user_id,))
            total_messages = cursor.fetchone()["total_messages"]
            
            cursor.execute("""
                SELECT COUNT(*) as total_sessions FROM sessions WHERE user_id = ?
            """, (user_id,))
            total_sessions = cursor.fetchone()["total_sessions"]
            
            return {
                "total_messages": total_messages,
                "total_sessions": total_sessions,
                "avg_messages_per_session": total_messages / max(total_sessions, 1)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def close(self):
        """Close database connection"""
        self.connection.close()
        logger.info("Database connection closed")
    
    def __del__(self):
        """Ensure database is closed"""
        try:
            self.close()
        except:
            pass
