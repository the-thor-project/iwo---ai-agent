# Quick Start Guide

## 1. Initial Setup (One-Time)

```bash
cd c:\Users\UserName\iwo---ai-agent/backend
python setup.py
cd ..
```

## 2. Choose Your Interface

### CLI (Best for Testing)
```bash
python cli/chatbot_cli.py
```

Then try:
```
You: Hello, how are you?
You: /help
You: /exit
```

### Web Interface (User-Friendly)
```bash
python web/app.py
```

Open: `http://localhost:8080` in your browser

## 3. Key Features

✓ Conversation History (persistent)
✓ Multiple Conversations (sessions)
✓ Python NLP Engine
✓ C System Operations
✓ Web & CLI Interfaces
✓ SQLite Database

## 4. File Structure Overview

```
ai-agent/
├── backend/        ← Python AI engine
├── c_extension/    ← C module for speed
├── web/            ← Web interface
├── cli/            ← Command-line tool
├── data/           ← Database (auto-created)
└── logs/           ← Log files (auto-created)
```

## 5. Common Commands

### CLI
- Type messages to chat
- `/new` → Start new conversation
- `/history` → See previous messages
- `/stats` → View statistics
- `/exit` → Quit

### Web
- Click "New Chat" to start
- Browse "Conversations" sidebar
- "Clear History" to delete
- Auto-saves everything

## 6. Configuration

Edit `config.json` to change:
- Web port (default: 8080)
- NLP model (default: distilbert)
- Database location
- Logging level

## 7. Troubleshooting

**Error: Module not found**
```bash
pip install -r backend/requirements.txt
```

**C extension won't build?**
- Windows: Install Visual C++ Build Tools
- macOS: `xcode-select --install`
- Linux: `sudo apt-get install build-essential python3-dev`

**Port 8080 in use?**
- Edit `config.json`, change `web.port` to 8081

## 8. Next Steps

1. ✓ Run setup.py
2. ✓ Start CLI or Web
3. ✓ Chat and test features
4. ✓ Explore code
5. ✓ Customize as needed

## Architecture Overview

```
┌─────────────────────────────────────┐
│      User + Interfaces              │
│  (CLI, Web, API)                    │
└────────────────┬────────────────────┘
                 │
        ┌────────▼────────┐
        │  Python Backend │
        │  • NLP Engine   │
        │  • API Server   │
        │  • Database Mgr │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐    ┌───▼──────┐  ┌──▼────┐
│ Data │    │ C Module │  │ Logs  │
│ Base │    │ (System) │  │       │
└──────┘    └──────────┘  └───────┘
```

## Running Everything

**Full Setup:**
```bash
# 1. Initial setup
python setup.py

# 2. Try CLI
python cli/chatbot_cli.py

# 3. Try Web (separate terminal)
python web/app.py
# Then visit http://localhost:8080
```

**That's it!** Your AI chatbot is ready to use. 🚀

---

For detailed info, see:
- `README.md` - Project overview
- `INSTALL.md` - Full installation guide
- `config.json` - Configuration options
