import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add the current directory to sys.path so that 'app' and 'nexova_core' are discoverable
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n" + "="*60)
    print("  GridSense AI — Modular Version Initializing")
    print("="*60)
    
    # Run the FastAPI app
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
