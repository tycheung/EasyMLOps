#!/usr/bin/env python3
"""
EasyMLOps Demo Script
A simple one-click way to start EasyMLOps in demo mode with SQLite
"""

import sys
import subprocess
import shutil
from pathlib import Path

def is_poetry_project():
    """Check if this is a Poetry project"""
    return Path("pyproject.toml").exists() and Path("poetry.lock").exists()

def get_python_cmd():
    """Get the appropriate Python command to use"""
    if is_poetry_project() and shutil.which("poetry"):
        return ["poetry", "run", "python"]
    else:
        return [sys.executable]

def main():
    """Start EasyMLOps in demo mode with SQLite"""
    print("🚀 EasyMLOps Demo Mode")
    print("=" * 50)
    print()
    print("Starting EasyMLOps with:")
    print("✅ SQLite database (no PostgreSQL required)")
    print("✅ Local file storage")
    print("✅ Demo data and examples")
    print("✅ Browser auto-open")
    print()
    
    # Get the appropriate Python command
    python_cmd = get_python_cmd()
    
    try:
        # Start the application in demo mode (main.py handles everything)
        cmd = python_cmd + ["-m", "app.main", "--demo"]
        
        print(f"🐍 Using: {' '.join(python_cmd)}")
        print("🌐 Will start at: http://localhost:8000")
        print("📖 API docs at: http://localhost:8000/docs")
        print()
        print("Press Ctrl+C to stop")
        print("-" * 50)
        print()
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except FileNotFoundError as e:
        print("❌ Error: Could not find the EasyMLOps application")
        print("Make sure you're in the EasyMLOps project directory and dependencies are installed")
        if is_poetry_project():
            print("💡 Try: poetry install")
        else:
            print("💡 Try: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 