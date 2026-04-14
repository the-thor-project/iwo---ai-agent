# INSTALLATION AND USAGE GUIDE

## System Requirements

- Python 3.8 or higher
- C compiler (for building C extension)
- 2GB RAM minimum
- Internet connection (for downloading models)

## Installation Steps

### 1. Clone/Extract the Project

Navigate to the project directory:
```bash
cd ai-agent
```

### 2. Run Initial Setup

Windows:
```powershell
python setup.py
```

Linux/macOS:
```bash
python3 setup.py
```

This will:
- Create necessary directories
- Install Python dependencies
- Build C extension module
- Initialize data files

### 3. Manual Dependency Installation (if needed)

```bash
pip install -r backend/requirements.txt
```

## Usage

### Option 1: Command-Line Interface (Recommended for Testing)

```bash
python cli/chatbot_cli.py
```

**CLI Commands:**
- `/help` - Show help
- `/clear` - Clear history
- `/history` - Show conversation history
- `/new` - Start new session
- `/stats` - Show statistics
- `/sessions` - List all sessions
- `/load <id>` - Load specific session
- `/export` - Export current session
- `/exit` - Exit application

### Option 2: Web Interface

Start the web server:
```bash
python web/app.py
```

Then open in your browser:
```
http://localhost:8080
```

### Option 3: Backend Testing

Direct Python testing:
```bash
python backend/app.py
```

## Project Architecture

```
┌─────────────────────────────────────────────┐
│           User Interfaces                   │
│  ┌──────────────────┬──────────────────┐   │
│  │   Web UI         │    CLI           │   │
│  │  (Flask)         │  (Interactive)   │   │
│  └────────┬─────────┴────────┬─────────┘   │
└───────────┼─────────────────────┼───────────┘
            │                     │
            └─────────┬───────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Core API (Python)       │
        │  ┌─────────────────────┐  │
        │  │   NLP Engine        │  │
        │  │ (transformers)      │  │
        │  └─────────────────────┘  │
        └─────────────┬──────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼─────┐         ┌────────▼───┐
    │ Database │         │ C Extension│
    │ (SQLite) │         │(system_ops)│
    └──────────┘         └────────────┘
```

## File Structure

```
ai-agent/
├── README.md                  # Project documentation
├── config.json               # Application configuration
├── setup.py                  # Setup script
├── .gitignore               # Git ignore rules
│
├── backend/
│   ├── app.py               # Main chatbot application
│   ├── nlp_engine.py        # NLP processing module
│   ├── database.py          # Database operations
│   ├── config.py            # Configuration handler
│   ├── requirements.txt      # Python dependencies
│   └── __init__.py          # Package init
│
├── c_extension/
│   ├── system_ops.c         # C implementation
│   ├── setup.py             # C extension builder
│   └── system_ops.so        # Compiled module (auto-generated)
│
├── web/
│   ├── app.py               # Flask web application
│   ├── __init__.py          # Package init
│   ├── templates/
│   │   └── index.html       # Web UI template
│   └── static/
│       ├── style.css        # Web UI styles
│       └── script.js        # Web UI client script
│
├── cli/
│   ├── chatbot_cli.py       # Command-line interface
│   └── __init__.py          # Package init
│
├── data/
│   └── conversations.db     # Auto-created SQLite database
│
└── logs/
    └── app.log              # Application logs (auto-created)
```

## Configuration

Edit `config.json` to customize:
- NLP model
- Database path
- Web server port
- Logging level
- API timeout settings

## Troubleshooting

### Issue: C Extension Build Fails

**Solution:**
- Windows: Install Visual C++ Build Tools
- Linux: `sudo apt-get install build-essential python3-dev`
- macOS: `xcode-select --install`

### Issue: Module Not Found Errors

**Solution:**
- Ensure you're in the project root directory
- Run: `pip install -r backend/requirements.txt`
- Restart Python kernel if in IDE

### Issue: Database Locked

**Solution:**
- Close other instances of the chatbot
- Delete `data/conversations.db` to start fresh

### Issue: Web Interface Not Loading

**Solution:**
- Check if port 8080 is available
- Try different port: Edit `config.json` and change `web.port`
- Check logs: `tail -f logs/app.log`

## Performance Tips

1. **NLP Model Selection**: Change model in `config.json` to a lighter one:
   - `distilbert-base-uncased` (default, faster)
   - `bert-base-uncased` (more accurate, slower)

2. **Database Optimization**:
   - Periodically backup and clean old data
   - Database is auto-indexed for performance

3. **Memory Management**:
   - Set appropriate `max_tokens` in config
   - Monitor memory usage with system monitor

## Development

### Adding New Features

1. Backend changes: Edit `backend/` modules
2. Web UI: Modify `web/templates/` and `web/static/`
3. CLI: Update `cli/chatbot_cli.py`
4. C operations: Edit `c_extension/system_ops.c`

### Testing

Run the test suite:
```bash
python backend/app.py
```

## API Reference

### /api/chat (POST)
Send message to chatbot
```json
{
  "message": "Hello",
  "session_id": "optional-session-id"
}
```

### /api/history (GET)
Get conversation history
```
/api/history?session_id=optional&limit=50
```

### /api/sessions (GET)
Get all user sessions

### /api/stats (GET)
Get user statistics

## Contributing

1. Follow PEP 8 code style
2. Add docstrings to functions
3. Test before submitting
4. Update documentation

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check this documentation
2. Review logs in `logs/` directory
3. Check GitHub issues
4. Submit detailed bug reports

---

Happy chatting! 🤖
