"""
Web Interface for AI Chatbot
Flask-based web server with RESTful API and web UI
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app import AIChatbot
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
web_app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(web_app)

# Configuration
web_app.config['SECRET_KEY'] = 'ai-chatbot-secret-key-change-in-production'
web_app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
web_app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Initialize chatbot
try:
    chatbot = AIChatbot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    chatbot = None


@web_app.before_request
def before_request():
    """Initialize user session"""
    session.permanent = True
    if 'user_id' not in session:
        import uuid
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"New user session created: {session['user_id']}")


@web_app.route('/')
def index():
    """Main chat page"""
    return render_template('index.html')


@web_app.route('/api/chat', methods=['POST'])
def chat():
    """Chat API endpoint"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        user_id = session.get('user_id')
        
        # Process message
        result = chatbot.process_message(user_id, user_message, session_id)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        user_id = session.get('user_id')
        session_id = request.args.get('session_id')
        limit = request.args.get('limit', 50, type=int)
        
        history = chatbot.get_conversation_history(user_id, session_id, limit)
        
        return jsonify({
            "status": "success",
            "history": history
        })
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions for user"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        user_id = session.get('user_id')
        sessions = chatbot.db.get_sessions(user_id)
        
        return jsonify({
            "status": "success",
            "sessions": sessions
        })
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/new-session', methods=['POST'])
def new_session():
    """Create new chat session"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        user_id = session.get('user_id')
        data = request.get_json() or {}
        title = data.get('title')
        
        session_id = chatbot.start_new_session(user_id)
        
        return jsonify({
            "status": "success",
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        user_id = session.get('user_id')
        session_id = request.args.get('session_id')
        
        success = chatbot.clear_history(user_id, session_id)
        
        return jsonify({
            "status": "success" if success else "error",
            "cleared": success
        })
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get user statistics"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 503
    
    try:
        user_id = session.get('user_id')
        stats = chatbot.db.get_statistics(user_id)
        
        return jsonify({
            "status": "success",
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        return jsonify({"error": str(e)}), 500


@web_app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": chatbot is not None
    })


@web_app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Not found"}), 404


@web_app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Run web server"""
    config = Config()
    logger.info(f"Starting {config.app_name} Web Interface")
    logger.info(f"Listening on http://localhost:{config.web_port}")
    
    web_app.run(
        host='127.0.0.1',
        port=config.web_port,
        debug=config.web_debug,
        threaded=True,
        use_reloader=False
    )


if __name__ == '__main__':
    main()
