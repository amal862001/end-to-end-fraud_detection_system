"""
Start Monitoring Stack

Convenience script to start the Prometheus + Grafana monitoring stack.

Author: Your Name
Date: 2026-01-15
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: Docker not found!")
        print("\n💡 Solution:")
        print("  Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        return False


def check_docker_compose():
    """Check if Docker Compose is installed."""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Docker Compose found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: Docker Compose not found!")
        print("\n💡 Solution:")
        print("  Docker Compose is included with Docker Desktop")
        return False


def start_monitoring_stack():
    """Start the monitoring stack with Docker Compose."""
    print("\n" + "="*60)
    print("STARTING MONITORING STACK")
    print("="*60)
    
    # Check prerequisites
    if not check_docker():
        return False
    
    if not check_docker_compose():
        return False
    
    # Change to monitoring directory
    monitoring_dir = Path(__file__).parent
    os.chdir(monitoring_dir)
    
    print("\n✓ Starting services...")
    print("  - Fraud Detection API (port 8000)")
    print("  - Prometheus (port 9090)")
    print("  - Grafana (port 3000)")
    print("  - Node Exporter (port 9100)")
    print("  - cAdvisor (port 8080)")
    print("  - Alertmanager (port 9093)")
    
    try:
        # Start Docker Compose
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True
        )
        
        print("\n✓ Services started successfully!")
        
        # Wait for services to be ready
        print("\n⏳ Waiting for services to be ready...")
        time.sleep(10)
        
        # Check service status
        print("\n📊 Service Status:")
        subprocess.run(["docker-compose", "ps"])
        
        print("\n" + "="*60)
        print("MONITORING STACK READY!")
        print("="*60)
        
        print("\n🌐 Access Points:")
        print("  - Grafana:     http://localhost:3000 (admin/admin)")
        print("  - Prometheus:  http://localhost:9090")
        print("  - API:         http://localhost:8000")
        print("  - API Metrics: http://localhost:8000/metrics")
        print("  - API Docs:    http://localhost:8000/docs")
        print("  - cAdvisor:    http://localhost:8080")
        print("  - Alertmanager: http://localhost:9093")
        
        print("\n📊 Next Steps:")
        print("  1. Open Grafana: http://localhost:3000")
        print("  2. Login with admin/admin")
        print("  3. Navigate to 'Fraud Detection API - Overview' dashboard")
        print("  4. Generate traffic: python monitoring/generate_traffic.py")
        
        print("\n🛑 To stop:")
        print("  cd monitoring && docker-compose down")
        
        print("\n📝 View logs:")
        print("  cd monitoring && docker-compose logs -f")
        
        print("\n" + "="*60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Failed to start services: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False


def stop_monitoring_stack():
    """Stop the monitoring stack."""
    print("\n" + "="*60)
    print("STOPPING MONITORING STACK")
    print("="*60)
    
    monitoring_dir = Path(__file__).parent
    os.chdir(monitoring_dir)
    
    try:
        subprocess.run(
            ["docker-compose", "down"],
            check=True
        )
        print("\n✓ Services stopped successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Failed to stop services: {e}")
        return False


def view_logs():
    """View logs from all services."""
    monitoring_dir = Path(__file__).parent
    os.chdir(monitoring_dir)
    
    try:
        subprocess.run(
            ["docker-compose", "logs", "-f"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ Stopped viewing logs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Monitoring Stack")
    parser.add_argument(
        "action",
        choices=["start", "stop", "logs", "status"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    if args.action == "start":
        start_monitoring_stack()
    elif args.action == "stop":
        stop_monitoring_stack()
    elif args.action == "logs":
        view_logs()
    elif args.action == "status":
        monitoring_dir = Path(__file__).parent
        os.chdir(monitoring_dir)
        subprocess.run(["docker-compose", "ps"])

