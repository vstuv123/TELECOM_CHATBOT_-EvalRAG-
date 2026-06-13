import os
import sys
import subprocess

def run_script(script_path):
    print(f"🚀 Running ingestion: {script_path}...")
    # Runs the python script and pipes output to console live
    result = subprocess.run([sys.executable, script_path], check=False)
    
    if result.returncode != 0:
        print(f"❌ Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ Finished: {script_path}\n")

if __name__ == "__main__":
    print("============================================================")
    # Target your specific files inside your ingests folder
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
