#!/bin/bash
BACKUP_DIR="$HOME/jarvis_ai/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="jarvis_backup_$DATE.tar.gz"

echo "Creating backup..."
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='jarvis_env' \
    --exclude='logs' \
    --exclude='backups' \
    "$HOME/jarvis_ai"

echo "Backup created: $BACKUP_FILE"

# Keep only last 10 backups
cd "$BACKUP_DIR"
ls -t jarvis_backup_*.tar.gz | tail -n +11 | xargs -r rm
