"""
Project Setup Script
Initialize and prepare the AI Chatbot project
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary project directories"""
    directories = [
        "data",
        "logs",
        "backend",
        "c_extension",
        "web/templates",
        "web/static",
        "cli"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ Python dependencies installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install Python dependencies")
        print("  Run manually: pip install -r backend/requirements.txt")


def build_c_extension():
    """Build C extension module"""
    print("\nBuilding C extension...")
    
    try:
        os.chdir("c_extension")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.chdir("..")
        print("✓ C extension built successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to build C extension")
        print("  Build environment may need C compiler")
        os.chdir("..")


def create_sample_data():
    """Create sample data files"""
    print("\nInitializing data files...")
    
    # Create empty logs directory
    Path("logs").mkdir(exist_ok=True)
    print("✓ Log directory initialized")
    
    # Create .gitignore
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        with open(gitignore, 'w') as f:
            f.write("""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project
data/conversations.db
logs/
*.log
cli_history.txt

# Build
build/
dist/
*.egg-info/

# OS
.DS_Store
Thumbs.db
""")
        print("✓ .gitignore created")


def print_summary():
    """Print setup summary"""
    print("\n" + "=" * 70)
    print("AI CHATBOT - SETUP COMPLETE")
    print("=" * 70)
    print("\nProject Structure:")
    print("  - backend/          : Python backend (NLP engine)")
    print("  - c_extension/      : C extension module")
    print("  - web/              : Flask web interface")
    print("  - cli/              : Command-line interface")
    print("  - data/             : Database and storage")
    print("  - logs/             : Application logs")
    print("\nNext Steps:")
    print("\n1. CLI Mode:")
    print("   python cli/chatbot_cli.py")
    print("\n2. Web Interface:")
    print("   python web/app.py")
    print("   Then open: http://localhost:8080")
    print("\n3. Backend Testing:")
    print("   python backend/app.py")
    print("\nFor more information, see README.md")
    print("=" * 70 + "\n")


def main():
    """Run setup"""
    print("\n" + "=" * 70)
    print("AI CHATBOT - PROJECT SETUP")
    print("=" * 70 + "\n")
    
    try:
        create_directories()
        install_python_dependencies()
        build_c_extension()
        create_sample_data()
        print_summary()
        
        print("Setup completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n✗ Setup failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
