#!/usr/bin/env python3
import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

# Add parent directory to path so we can import from data directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run the Fan Intelligence Dashboard."""
    print("🚀 Starting Fan Intelligence Dashboard MVP+...")
    
    # Check if data files exist
    data_dir = Path("../data")
    required_files = [
        data_dir / "messages_with_emotions.pkl"
    ]
    
    for file in required_files:
        if not file.exists():
            print(f"❌ Error: Required data file not found: {file}")
            print("Please ensure the data files are in the correct location.")
            return
    
    # Start the FastAPI backend
    print("\n📡 Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)
    
    # Open browser
    dashboard_url = "http://localhost:8000"
    print(f"\n✅ Dashboard is ready! Opening {dashboard_url} in your browser...")
    webbrowser.open(dashboard_url)
    
    print("\n📊 Fan Intelligence Dashboard is running!")
    print("   Dashboard: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the dashboard.\n")
    
    try:
        # Keep the script running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down dashboard...")
        backend_process.terminate()
        print("✅ Dashboard stopped successfully.")

if __name__ == "__main__":
    main()