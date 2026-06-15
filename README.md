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
## bugs
example to bug:
```iwo
You: who build you?
                                                                                                                        Bot: 2026-06-15 18:24:46,654 - nlp_engine - INFO - IWO generated advanced response for intent: question (confidence: 0.8
4, style=concise)
2026-06-15 18:24:46,657 - app - INFO - Message processed for user cli_user
I understand your question perfectly. Here's what I can tell you:                                                       
You: and you can tell me...

Bot: 2026-06-15 18:25:13,330 - nlp_engine - INFO - IWO generated advanced response for intent: default (confidence: 0.50
, style=friendly)
2026-06-15 18:25:13,334 - app - INFO - Message processed for user cli_user                                              using you ai you task task task task ai task ai ai task task ai task ai ai task task ai task task you task task ai task
task ai task ai ai math task task task task task task a task ai task ai ai task ai task task ai ai task task task task t
ask ai task task task task task api                                                                                                                                                                                                             Anything else you want to explore? ✨

You: no

Bot: 2026-06-15 18:25:36,165 - nlp_engine - INFO - IWO generated advanced response for intent: default (confidence: 0.50
, style=friendly)                                                                                                       2026-06-15 18:25:36,169 - app - INFO - Message processed for user cli_user
I appreciate you sharing that. Let me offer some thoughts.
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
- Flask
- SQLite
- C (system operations)
