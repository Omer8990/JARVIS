#!/bin/bash
cd "$(dirname "$0")/.."
echo "Updating Jarvis..."

# Stop service
sudo systemctl stop jarvis

# Backup current version
./scripts/backup_jarvis.sh

# Update Python packages
source jarvis_env/bin/activate
pip install --upgrade -r requirements.txt

# Update Ollama models
ollama pull llama3.2:3b

# Restart service
sudo systemctl start jarvis

echo "Jarvis updated successfully!"
