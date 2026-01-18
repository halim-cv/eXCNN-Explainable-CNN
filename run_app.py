import subprocess
import sys
import time
import os
import signal

def run_app():
    """
    Starts both the Backend (FastAPI/Uvicorn) and Frontend (SimpleHTTPServer) 
    as concurrent subprocesses.
    """
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_root, 'frontend')

    print(">>> Starting eXCNN Platform...")
    print("--------------------------------")
    
    # Start Backend (Uvicorn)
    # We use sys.executable to ensure we use the same python interpreter
    print("   [1/2] Starting Backend API (port 8000)...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.api:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    try:
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=project_root,
        )
    except Exception as e:
        print(f"[ERROR] Failed to start backend: {e}")
        return

    # Wait a moment for backend to initialize
    time.sleep(2)

    # Start Frontend (HTTP Server)
    print("   [2/2] Starting Frontend Server (port 8080)...")
    frontend_cmd = [
        sys.executable, "-m", "http.server", 
        "8080", 
        "--directory", frontend_dir
    ]

    try:
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=project_root
        )
    except Exception as e:
        print(f"[ERROR] Failed to start frontend: {e}")
        backend_process.terminate()
        return

    print("\n[OK] Application is fully running!")
    print("   -> Open Browser: http://localhost:8080")
    print("   Backend API:    http://localhost:8000")
    print("\nPress Ctrl+C to stop both servers.\n")

    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_process.poll() is not None:
                print("\n[ERROR] Backend process exited unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("\n[ERROR] Frontend process exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n[STOP] Stopping servers...")
    finally:
        # Graceful shutdown
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for them to exit
        backend_process.wait()
        frontend_process.wait()
        print("Servers stopped. Goodbye!")

if __name__ == "__main__":
    run_app()
