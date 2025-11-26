"""
Setup script for CrowdVision development environment
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    """Main setup function"""
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("CrowdVision Development Environment Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    # Create virtual environment
    print("\n1. Creating virtual environment...")
    venv_path = project_root / "venv"
    if not venv_path.exists():
        if not run_command(f"{sys.executable} -m venv venv", cwd=project_root):
            print("Failed to create virtual environment")
            sys.exit(1)
    else:
        print("Virtual environment already exists")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("\n2. Upgrading pip...")
    run_command(f"{pip_path} install --upgrade pip", cwd=project_root)
    
    # Install requirements
    print("\n3. Installing dependencies...")
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        if not run_command(f"{pip_path} install -r requirements.txt", cwd=project_root):
            print("Failed to install dependencies")
            sys.exit(1)
    else:
        print("Warning: requirements.txt not found")
    
    # Create necessary directories
    print("\n4. Creating directory structure...")
    directories = [
        "models/checkpoints",
        "data/raw",
        "data/processed",
        "data/datasets",
        "logs",
        "chroma_db"
    ]
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    # Copy .env.example to .env if not exists
    print("\n5. Setting up environment variables...")
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("Created .env file from .env.example")
        print("Please edit .env with your API keys and configuration")
    else:
        print(".env file already exists")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Edit .env with your configuration")
    print("3. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000")
    print("4. Run tests: pytest tests/")
    print("5. Start API: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
