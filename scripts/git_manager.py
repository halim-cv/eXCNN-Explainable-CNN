import os
import subprocess
import sys

POSSIBLE_PATHS = [
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
    r"C:\Users\Sam-tech\AppData\Local\Programs\Git\cmd\git.exe",
    r"C:\Users\Sam-tech\AppData\Local\Programs\Git\bin\git.exe",
]

def find_git():
    # Check PATH first (though we suspect it fails)
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "git"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            return path
    return None

def run_git_cmd(git_path, args, cwd):
    cmd = [git_path] + args
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Execution error: {e}")
        return False

def main():
    # Project root is parent of scripts/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_path = find_git()
    
    if not git_path:
        print("❌ Could not find git.exe in common locations.")
        print("Please verify Git is installed.")
        return

    print(f"✅ Found git at: {git_path}")
    
    # Check if .git exists
    if not os.path.exists(os.path.join(root, ".git")):
        print("Initializing git repo...")
        run_git_cmd(git_path, ["init"], root)
    
    # Check status
    print("Checking status...")
    run_git_cmd(git_path, ["status"], root)

    # Add files
    print("Adding all files...")
    run_git_cmd(git_path, ["add", "."], root)
    
    # Commit
    print("Committing...")
    run_git_cmd(git_path, ["commit", "-m", "Update: Complete redesign (Monochrome), cleanup scripts, full ImageNet labels"], root)
    
    # Push
    print("Pushing to origin...")
    # Check for remote
    res = subprocess.run([git_path, "remote", "-v"], cwd=root, text=True, capture_output=True)
    if "origin" not in res.stdout:
        print("❌ No remote 'origin' found. Cannot push.")
        print("Please run: git remote add origin <YOUR_REPO_URL>")
        return

    success = run_git_cmd(git_path, ["push", "origin", "main"], root)
    if not success:
         print("Trying push to master...")
         run_git_cmd(git_path, ["push", "origin", "master"], root)

if __name__ == "__main__":
    main()
