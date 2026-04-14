"""
Main AI Chatbot Application
Integrates Python NLP engine with C system operations
"""

import json
import logging
from pathlib import Path
from database import Database
from nlp_engine import NLPEngine
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIChatbot:
    """Main chatbot orchestrator"""
    
    def __init__(self):
        """Initialize chatbot with configuration"""
        self.config = Config()
        self.db = Database(self.config.database_path)
        self.nlp_engine = NLPEngine(self.config)
        logger.info("AI Chatbot initialized")
    
    def process_message(self, user_id: str, message: str, session_id: str = None):
        """
        Process user message and generate response
        
        Args:
            user_id: User identifier
            message: User message text
            session_id: Conversation session ID
            
        Returns:
            dict: Response with generated text and metadata
        """
        try:
            # Generate AI response
            response = self.nlp_engine.generate_response(message)
            
            # Store conversation
            session_id = self.db.save_message(
                user_id=user_id,
                message=message,
                response=response,
                session_id=session_id
            )
            
            logger.info(f"Message processed for user {user_id}")
            
            return {
                "status": "success",
                "response": response,
                "session_id": session_id,
                "timestamp": self.db.get_timestamp()
            }
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": "I encountered an error processing your message."
            }
    
    def get_conversation_history(self, user_id: str, session_id: str = None, limit: int = 50):
        """
        Retrieve conversation history
        
        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Number of messages to retrieve
            
        Returns:
            list: Conversation history
        """
        return self.db.get_messages(user_id, session_id, limit)
    
    def start_new_session(self, user_id: str):
        """Start a new conversation session"""
        return self.db.create_session(user_id)
    
    def clear_history(self, user_id: str, session_id: str = None):
        """Clear conversation history"""
        return self.db.delete_messages(user_id, session_id)
    
    def shutdown(self):
        """Gracefully shutdown chatbot"""
        self.db.close()
        logger.info("AI Chatbot shutdown complete")


def main():
    """Test the chatbot"""
    chatbot = AIChatbot()
    
    # Test conversation
    user_id = "test_user_123"
    session_id = chatbot.start_new_session(user_id)
    
    print("=" * 60)
    print("AI Chatbot - Test Mode")
    print("=" * 60)
    
    test_messages = [
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me about Python programming"
    ]
    
    for msg in test_messages:
        print(f"\nUser: {msg}")
        result = chatbot.process_message(user_id, msg, session_id)
        if result["status"] == "success":
            print(f"Bot: {result['response']}")
        else:
            print(f"Error: {result['error']}")
    
    # Show history
    print("\n" + "=" * 60)
    print("Conversation History:")
    print("=" * 60)
    history = chatbot.get_conversation_history(user_id, session_id)
    for msg in history:
        print(f"User: {msg['user_message']}")
        print(f"Bot: {msg['bot_response']}\n")
    
    chatbot.shutdown()


if __name__ == "__main__":
    main()
