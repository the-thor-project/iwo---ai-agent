"""
Command-Line Interface for AI Chatbot
Provides an interactive CLI for chatting with the AI
"""

import sys
from pathlib import Path
import readline  # For better input handling
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app import AIChatbot
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotCLI:
    """Command-line interface for the chatbot"""
    
    def __init__(self):
        """Initialize CLI"""
        self.config = Config()
        self.chatbot = AIChatbot()
        self.user_id = "cli_user"
        self.session_id = None
        self.history = []
        self.running = True
        
        # Create user
        self.chatbot.db.create_user(self.user_id)
        
        logger.info("CLI Chatbot initialized")
    
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "=" * 70)
        print("  AI CHATBOT - Command Line Interface")
        print("=" * 70)
        print(f"  Version: {self.config.app_version}")
        print(f"  Type 'help' for commands, 'exit' to quit")
        print("=" * 70 + "\n")
    
    def display_help(self):
        """Display help message"""
        print("""
Available Commands:
  help           - Show this help message
  clear          - Clear conversation history
  history        - Show conversation history
  new            - Start a new conversation
  stats          - Show statistics
  sessions       - List all sessions
  load <id>      - Load a specific session
  export         - Export current session
  exit           - Exit the application
  
Just type your message to chat with the AI.
""")
    
    def run(self):
        """Main CLI loop"""
        self.display_banner()
        self.create_new_session()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input("\nYou: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower().startswith("/"):
                        self.handle_command(user_input[1:])
                    else:
                        self.process_message(user_input)
                
                except KeyboardInterrupt:
                    print("\n[Interrupted]")
                except EOFError:
                    break
        
        finally:
            self.shutdown()
    
    def handle_command(self, command):
        """Handle CLI commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        
        if cmd in ["help", "h", "?"]:
            self.display_help()
        elif cmd in ["exit", "quit", "q"]:
            self.running = False
            print("\nGoodbye!")
        elif cmd in ["clear", "c"]:
            self.clear_history()
        elif cmd in ["history", "hist"]:
            self.show_history()
        elif cmd in ["new", "n"]:
            self.create_new_session()
        elif cmd in ["stats", "stat"]:
            self.show_stats()
        elif cmd in ["sessions", "sess"]:
            self.list_sessions()
        elif cmd == "load" and args:
            self.load_session(args[0])
        elif cmd == "export":
            self.export_session()
        else:
            print(f"Unknown command: /{cmd}. Type 'help' for available commands.")
    
    def process_message(self, user_input):
        """Process user message"""
        print("\nBot: ", end="", flush=True)
        
        try:
            result = self.chatbot.process_message(
                self.user_id,
                user_input,
                self.session_id
            )
            
            if result["status"] == "success":
                print(result["response"])
                self.session_id = result["session_id"]
                self.history.append({
                    "user": user_input,
                    "bot": result["response"],
                    "timestamp": result["timestamp"]
                })
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            print(f"Error: {str(e)}")
    
    def create_new_session(self):
        """Create a new conversation session"""
        try:
            self.session_id = self.chatbot.start_new_session(self.user_id)
            self.history = []
            print("\n[New conversation started]")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            print(f"Error creating session: {str(e)}")
    
    def clear_history(self):
        """Clear conversation history"""
        try:
            self.chatbot.clear_history(self.user_id, self.session_id)
            self.history = []
            print("[History cleared]")
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            print(f"Error: {str(e)}")
    
    def show_history(self):
        """Display conversation history"""
        if not self.history:
            print("[No messages in this session]")
            return
        
        print("\n" + "=" * 70)
        print("CONVERSATION HISTORY")
        print("=" * 70)
        
        for i, msg in enumerate(self.history, 1):
            print(f"\n[{i}] {msg['timestamp']}")
            print(f"You: {msg['user']}")
            print(f"Bot: {msg['bot']}")
        
        print("\n" + "=" * 70)
    
    def show_stats(self):
        """Display statistics"""
        try:
            stats = self.chatbot.db.get_statistics(self.user_id)
            
            print("\n" + "=" * 70)
            print("STATISTICS")
            print("=" * 70)
            print(f"Total Messages:        {stats.get('total_messages', 0)}")
            print(f"Total Sessions:        {stats.get('total_sessions', 0)}")
            print(f"Avg Messages/Session:  {stats.get('avg_messages_per_session', 0):.1f}")
            print("=" * 70)
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            print(f"Error: {str(e)}")
    
    def list_sessions(self):
        """List all sessions"""
        try:
            sessions = self.chatbot.db.get_sessions(self.user_id)
            
            if not sessions:
                print("[No sessions found]")
                return
            
            print("\n" + "=" * 70)
            print("SESSIONS")
            print("=" * 70)
            
            for i, session in enumerate(sessions, 1):
                marker = "[Current]" if session["session_id"] == self.session_id else "          "
                print(f"{marker} {i}. {session['title']}")
                print(f"   ID: {session['session_id']}")
                print(f"   Created: {session['created_at']}")
            
            print("=" * 70)
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            print(f"Error: {str(e)}")
    
    def load_session(self, session_id):
        """Load a specific session"""
        try:
            messages = self.chatbot.get_conversation_history(
                self.user_id,
                session_id
            )
            
            if not messages:
                print("[Session not found or is empty]")
                return
            
            self.session_id = session_id
            self.history = []
            
            print(f"\n[Loaded session: {session_id}]")
            print("\nRecent messages:")
            
            for msg in messages[-5:]:  # Show last 5 messages
                print(f"\nYou: {msg['user_message']}")
                print(f"Bot: {msg['bot_response']}")
        
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            print(f"Error: {str(e)}")
    
    def export_session(self):
        """Export current session to file"""
        try:
            if not self.history:
                print("[Nothing to export]")
                return
            
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = Path("data") / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("AI CHATBOT SESSION EXPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write("=" * 70 + "\n\n")
                
                for msg in self.history:
                    f.write(f"[{msg['timestamp']}]\n")
                    f.write(f"You: {msg['user']}\n")
                    f.write(f"Bot: {msg['bot']}\n")
                    f.write("-" * 70 + "\n\n")
            
            print(f"[Session exported to {filepath}]")
        
        except Exception as e:
            logger.error(f"Error exporting session: {str(e)}")
            print(f"Error: {str(e)}")
    
    def shutdown(self):
        """Gracefully shutdown CLI"""
        try:
            self.chatbot.shutdown()
            logger.info("CLI shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


def main():
    """Entry point for CLI"""
    cli = ChatbotCLI()
    cli.run()


if __name__ == "__main__":
    main()
