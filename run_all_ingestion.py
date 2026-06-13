import os
import sys
import subprocess

def run_script(script_path):
    print(f"🚀 Running ingestion: {script_path}...")
    
    # FIX: Inject the root workspace path into the child process environment
    current_env = os.environ.copy()
    root_dir = os.getcwd()
    
    # This forces the script to look at the root directory for 'models'
    if "PYTHONPATH" in current_env:
        current_env["PYTHONPATH"] = f"{root_dir}{os.pathsep}{current_env['PYTHONPATH']}"
    else:
        current_env["PYTHONPATH"] = root_dir

    result = subprocess.run([sys.executable, script_path], env=current_env, check=False)
    
    if result.returncode != 0:
        print(f"❌ Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ Finished: {script_path}\n")

if __name__ == "__main__":
    print("============================================================")
    ingestion_scripts = [
        "ingests/ingest_faq.py",
        "ingests/ingest_pdf.py",
        "ingests/ingest_tickets.py"
    ]
    
    for script in ingestion_scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"❌ Configuration Error: Script file not found at path: {script}")
            sys.exit(1)
            
    print("🎉 All three collections successfully built inside ChromaDB!")
    print("============================================================")
