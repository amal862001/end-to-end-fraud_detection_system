"""
Start FastAPI Server

Convenience script to start the fraud detection API.

Author: Your Name
Date: 2026-01-15
"""

import os
import sys
import subprocess
from pathlib import Path

def check_artifacts():
    """Check if model artifacts exist."""
    # Try both relative paths (from api/ and from project root)
    possible_paths = [
        Path('../artifacts'),
        Path('artifacts'),
        Path(__file__).parent.parent / 'artifacts'
    ]

    artifacts_dir = None
    for path in possible_paths:
        if path.exists():
            artifacts_dir = path
            break

    if artifacts_dir is None:
        print("❌ ERROR: artifacts/ directory not found!")
        print("\n💡 Solution:")
        print("  Run model serialization first:")
        print("  python src/models/serialize_model.py")
        return False

    required_files = [
        'fraud_model.pkl',
        'scaler.pkl',
        'threshold.txt'
    ]

    missing = []
    for file in required_files:
        if not (artifacts_dir / file).exists():
            missing.append(file)

    if missing:
        print("❌ ERROR: Missing model artifacts!")
        print("\nMissing files:")
        for file in missing:
            print(f"  - artifacts/{file}")
        print("\n💡 Solution:")
        print("  Run model serialization first:")
        print("  python src/models/serialize_model.py")
        return False

    print(f"✓ All model artifacts found in {artifacts_dir}")
    return True


def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server."""
    print("\n" + "="*60)
    print("STARTING FRAUD DETECTION API")
    print("="*60)
    
    # Check artifacts
    if not check_artifacts():
        return
    
    print(f"\n✓ Starting server at http://{host}:{port}")
    print(f"✓ Reload: {reload}")
    print("\n📚 Documentation:")
    print(f"  - Swagger UI: http://{host}:{port}/docs")
    print(f"  - ReDoc: http://{host}:{port}/redoc")
    print("\n🔍 Endpoints:")
    print(f"  - Health: http://{host}:{port}/health")
    print(f"  - Predict: http://{host}:{port}/predict")
    print(f"  - Batch: http://{host}:{port}/predict/batch")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start uvicorn
    cmd = [
        "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port)
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except FileNotFoundError:
        print("\n❌ ERROR: uvicorn not found!")
        print("\n💡 Solution:")
        print("  Install dependencies:")
        print("  pip install -r api/requirements.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Fraud Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )

