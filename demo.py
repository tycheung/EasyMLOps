#!/usr/bin/env python3
"""
EasyMLOps Demo Script
A simple one-click way to start EasyMLOps in demo mode with SQLite

This demo provides full access to all EasyMLOps features:
- Model deployment and management
- Schema validation and versioning
- Comprehensive monitoring (21 monitoring services)
- A/B testing and canary deployments
- Drift detection and performance degradation monitoring
- Bias & fairness monitoring
- Model explainability
- And much more!
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
    print("=" * 60)
    print()
    print("Starting EasyMLOps with:")
    print("✅ SQLite database (no PostgreSQL required)")
    print("✅ Auto-created directories (models, bentos, logs)")
    print("✅ Full feature access - all 21 monitoring services")
    print("✅ Browser auto-open to API documentation")
    print("✅ Interactive web interface at /")
    print()
    print("Available Features:")
    print("  📊 Model Deployment & Management")
    print("  🛡️  Schema Validation & Versioning")
    print("  📈 Performance Monitoring & Metrics")
    print("  🌊 Drift Detection (Feature, Data, Prediction)")
    print("  📉 Performance Degradation Detection")
    print("  🧪 A/B Testing Framework")
    print("  🦅 Canary Deployments")
    print("  ⚖️  Bias & Fairness Monitoring")
    print("  🔬 Model Explainability (SHAP, LIME)")
    print("  ✅ Data Quality & Anomaly Detection")
    print("  📋 Governance & Compliance")
    print("  📊 Analytics Dashboard")
    print()
    
    # Get the appropriate Python command
    python_cmd = get_python_cmd()
    
    try:
        # Start the application in demo mode (main.py handles everything)
        cmd = python_cmd + ["-m", "app.main", "--demo"]
        
        print(f"🐍 Using: {' '.join(python_cmd)}")
        print("🌐 Web Interface: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("📋 Alternative Docs: http://localhost:8000/redoc")
        print("💓 Health Check: http://localhost:8000/health")
        print()
        print("💡 Tips:")
        print("  - Upload a model file (.pkl, .joblib, .h5, etc.) via the web UI")
        print("  - Define input/output schemas for validation")
        print("  - Deploy models and test predictions")
        print("  - Explore monitoring features in the API docs")
        print()
        print("Press Ctrl+C to stop")
        print("-" * 60)
        print()
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
        print("Thanks for trying EasyMLOps!")
    except FileNotFoundError as e:
        print("❌ Error: Could not find the EasyMLOps application")
        print("Make sure you're in the EasyMLOps project directory and dependencies are installed")
        print()
        if is_poetry_project():
            print("💡 Installation:")
            print("   poetry install")
            print()
            print("💡 Then run:")
            print("   python demo.py")
        else:
            print("💡 Installation:")
            print("   pip install -r requirements.txt")
            print()
            print("💡 Then run:")
            print("   python demo.py")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting demo (exit code: {e.returncode})")
        print("Check the error messages above for details")
        return 1
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure all dependencies are installed")
        if is_poetry_project():
            print("     Run: poetry install")
        else:
            print("     Run: pip install -r requirements.txt")
        print("  2. Check that you're in the EasyMLOps project directory")
        print("  3. Verify Python 3.12+ is installed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 