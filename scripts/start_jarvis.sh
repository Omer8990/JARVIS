#!/bin/bash
cd "$(dirname "$0")/.."
source jarvis_env/bin/activate
python jarvis_agent.py
