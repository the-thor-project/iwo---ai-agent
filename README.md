# iwo - LLM-local System

A hybrid Python and C AI chatbot system with NLP capabilities, conversation history, web interface, and CLI.

## Architecture

- **Python Backend**: NLP/AI processing using transformers and language models
- **C Extension**: System operations and performance-critical tasks
- **Database**: SQLite for conversation history
- **Interfaces**: Web UI (Flask) and CLI

## Project Structure

```
ai-agent/
├── backend/              # Python backend code
│   ├── app.py           # Main application
│   ├── nlp_engine.py    # NLP processing
│   ├── database.py      # Database operations
│   ├── config.py        # Configuration
│   └── requirements.txt # Dependencies
├── c_extension/         # C extension module
│   ├── system_ops.c     # C implementation
│   ├── system_ops.h     # C header
│   └── setup.py         # Build configuration
├── web/                 # Web interface
│   ├── app.py          # Flask web app
│   ├── static/         # CSS/JS
│   └── templates/      # HTML templates
├── cli/                # CLI interface
│   └── chatbot_cli.py  # CLI application
├── data/               # Data storage
│   └── conversations.db # SQLite database
└── config.json         # Application config
```

## Installation & Setup

1. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. Build C extension:
   ```bash
   cd c_extension && python setup.py build_ext --inplace
   ```

3. Run as CLI:
   ```bash
   python cli/chatbot_cli.py
   ```

4. Run web interface:
   ```bash
   python web/app.py
   ```

## Features

- Conversational AI powered by transformers
- Conversation history and persistence
- Web-based chat interface
- Command-line interface
- C extension for optimized operations
- Session management
- Multi-user support ready

## Technology Stack

- Python 3.8+
- PyTorch/Transformers
- Flask
- SQLite
- C (system operations)
