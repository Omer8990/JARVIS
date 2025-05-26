#!/bin/bash
echo "=== Jarvis Status ==="
sudo systemctl status jarvis --no-pager
echo ""
echo "=== Ollama Status ==="
sudo systemctl status ollama --no-pager
echo ""
echo "=== System Resources ==="
free -h
df -h /
