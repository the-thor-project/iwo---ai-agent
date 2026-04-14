// AI Chatbot Web Interface - Client Script

class ChatbotUI {
    constructor() {
        this.currentSessionId = null;
        this.messages = [];
        this.isLoading = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.createNewSession();
        this.loadStats();
    }
    
    initializeElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.messageForm = document.getElementById('messageForm');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.sessionsList = document.getElementById('sessionsList');
        this.sessionTitle = document.getElementById('sessionTitle');
        this.statsDisplay = document.getElementById('statsDisplay');
        this.tokenCount = document.getElementById('tokenCount');
    }
    
    attachEventListeners() {
        this.messageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        this.newChatBtn.addEventListener('click', () => {
            this.createNewSession();
        });
        
        this.clearBtn.addEventListener('click', () => {
            if (confirm('Clear all conversation history?')) {
                this.clearHistory();
            }
        });
    }
    
    createNewSession() {
        fetch('/api/new-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                this.currentSessionId = data.session_id;
                this.messages = [];
                this.clearMessagesDisplay();
                this.loadSessions();
                this.messageInput.focus();
            }
        })
        .catch(error => console.error('Error creating session:', error));
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) return;
        
        this.isLoading = true;
        this.sendBtn.disabled = true;
        
        // Add user message to UI
        this.addMessage('user', message);
        this.messageInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Send to backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: this.currentSessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            this.removeTypingIndicator();
            
            if (data.status === 'success') {
                this.addMessage('bot', data.response);
                this.currentSessionId = data.session_id;
                
                if (data.tokens_used) {
                    this.updateTokenCount(data.tokens_used);
                }
            } else {
                this.addMessage('bot', `Error: ${data.error || 'Unknown error'}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.removeTypingIndicator();
            this.addMessage('bot', 'Connection error. Please try again.');
        })
        .finally(() => {
            this.isLoading = false;
            this.sendBtn.disabled = false;
            this.messageInput.focus();
        });
    }
    
    addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content typing-indicator';
        contentDiv.innerHTML = '<span></span><span></span><span></span>';
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    clearMessagesDisplay() {
        this.messagesContainer.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to AI Chatbot</h2>
                <p>Start a conversation by typing your message below.</p>
            </div>
        `;
        this.messagesContainer.scrollTop = 0;
    }
    
    loadSessions() {
        fetch('/api/sessions')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    this.displaySessions(data.sessions);
                }
            })
            .catch(error => console.error('Error loading sessions:', error));
    }
    
    displaySessions(sessions) {
        this.sessionsList.innerHTML = '';
        
        sessions.forEach(session => {
            const li = document.createElement('li');
            li.textContent = session.title;
            li.className = session.session_id === this.currentSessionId ? 'active' : '';
            li.addEventListener('click', () => {
                this.loadSession(session.session_id);
            });
            
            this.sessionsList.appendChild(li);
        });
    }
    
    loadSession(sessionId) {
        this.currentSessionId = sessionId;
        this.clearMessagesDisplay();
        
        fetch(`/api/history?session_id=${sessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    this.messages = data.history;
                    this.displayMessages();
                    this.loadSessions();
                }
            })
            .catch(error => console.error('Error loading session:', error));
    }
    
    displayMessages() {
        this.clearMessagesDisplay();
        
        this.messages.forEach(msg => {
            this.addMessage('user', msg.user_message);
            this.addMessage('bot', msg.bot_response);
        });
    }
    
    clearHistory() {
        fetch(`/api/clear?session_id=${this.currentSessionId}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.cleared) {
                this.clearMessagesDisplay();
                this.loadSessions();
                this.loadStats();
                alert('History cleared successfully');
            }
        })
        .catch(error => console.error('Error clearing history:', error));
    }
    
    loadStats() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const stats = data.stats;
                    this.statsDisplay.innerHTML = `
                        <div>Messages: ${stats.total_messages}</div>
                        <div>Sessions: ${stats.total_sessions}</div>
                    `;
                }
            })
            .catch(error => console.error('Error loading stats:', error));
    }
    
    updateTokenCount(tokens) {
        this.tokenCount.textContent = `Tokens: ${tokens}`;
    }
}

// Initialize UI when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatbotUI = new ChatbotUI();
});

// Keep alive and update stats periodically
setInterval(() => {
    if (window.chatbotUI) {
        window.chatbotUI.loadStats();
    }
}, 30000);
