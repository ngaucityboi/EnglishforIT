#!/usr/bin/env python
"""
Quick start script for Vietnamese Legal Assistant Demo
Runs the Desktop GUI app (PySimpleGUI)
"""

import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    os.chdir(r"L:\Download\EnglishforIT")
    
    print("=" * 50)
    print("Vietnamese Legal Assistant - Desktop App")
    print("=" * 50)
    print()
    print("Starting desktop app...")
    print()
    
    try:
        # Run desktop app
        subprocess.run([
            sys.executable,
            "step/5_demo/desktop_app.py"
        ])
    except KeyboardInterrupt:
        print("\n\nApp stopped.")
        print("Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
