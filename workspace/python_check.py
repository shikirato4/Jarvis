#!/usr/bin/env python3
import sys
import subprocess

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check if pip is available
try:
    result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=True)
    print(f"pip version: {result.stdout.strip()}")
except Exception as e:
    print(f"pip not found: {e}")

print("\nPython is installed and ready to use!")