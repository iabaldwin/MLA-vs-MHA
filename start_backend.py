#!/usr/bin/env python3
"""
Start the MLA vs MHA backend server
"""
import subprocess
import sys
import os

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting backend server...")
    print("Backend will be available at: http://localhost:8020")
    print("API docs available at: http://localhost:8020/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8020", "--reload"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("backend.py"):
        print("❌ backend.py not found. Please run this script from the transformers directory.")
        sys.exit(1)
    
    if not os.path.exists("mla/transformer.py"):
        print("❌ mla/transformer.py not found. Please ensure the transformer module is available.")
        sys.exit(1)
    
    # Install requirements if needed
    if not os.path.exists("requirements-backend.txt"):
        print("❌ requirements-backend.txt not found")
        sys.exit(1)
    
    start_server()
