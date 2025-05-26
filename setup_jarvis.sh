#!/bin/bash

# =================================================================
# JARVIS AI AGENT - PRODUCTION SETUP SCRIPT
# Advanced AI Assistant for Raspberry Pi with Hybrid Cloud/Local AI
# =================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
print_banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║                                                              ║"
    echo "  ║      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                ║"
    echo "  ║      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                ║"
    echo "  ║      ██║███████║██████╔╝██║   ██║██║███████╗                ║"
    echo "  ║ ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                ║"
    echo "  ║ ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                ║"
    echo "  ║  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                ║"
    echo "  ║                                                              ║"
    echo "  ║           Advanced AI Assistant Setup v2.0                   ║"
    echo "  ║              Raspberry Pi Production Ready                   ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Please don't run this script as root. We'll ask for sudo when needed."
        exit 1
    fi
}

# Check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check if running on Raspberry Pi
    if [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        log "Detected Raspberry Pi - Perfect for Jarvis!"
        PI_MODEL=$(cat /proc/device-tree/model)
        log_info "Model: $PI_MODEL"
    else
        log_warn "Not running on Raspberry Pi, but that's okay! Jarvis will work on most Linux systems."
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        log "Python version: $PYTHON_VERSION"
        
        # Check if Python 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log "Python version is compatible ✓"
        else
            log_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python3 is not installed!"
        exit 1
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    if [[ $TOTAL_MEM -lt 1024 ]]; then
        log_warn "Low memory detected (${TOTAL_MEM}MB). Jarvis will work but may be slower."
    else
        log "Memory: ${TOTAL_MEM}MB ✓"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ ${AVAILABLE_SPACE%.*} -lt 2 ]]; then
        log_warn "Low disk space. Ensure you have at least 2GB free for optimal performance."
    fi
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install required packages
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        portaudio19-dev \
        python3-pyaudio \
        alsa-utils \
        espeak \
        espeak-data \
        libespeak1 \
        libespeak-dev \
        ffmpeg \
        sqlite3 \
        curl \
        wget \
        git \
        htop \
        screen \
        ufw \
        fail2ban
    
    log "System dependencies installed ✓"
}

# Test audio system
test_audio() {
    log "Testing audio system..."
    
    # Test speakers
    log_info "Testing text-to-speech..."
    if command -v espeak &> /dev/null; then
        espeak "Audio test successful" 2>/dev/null || log_warn "Speaker test failed - check audio output"
    fi
    
    # List audio devices
    log_info "Available audio devices:"
    python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'  Device {i}: {info[\"name\"]} - {info[\"maxInputChannels\"]} in, {info[\"maxOutputChannels\"]} out')
p.terminate()
" 2>/dev/null || log_warn "Could not list audio devices"
}

# Create project structure
create_project_structure() {
    log "Creating project structure..."
    
    # Create main directory
    JARVIS_DIR="$HOME/jarvis_ai"
    mkdir -p "$JARVIS_DIR"
    cp jarvis_agent.py "$JARVIS_DIR"
    cd "$JARVIS_DIR"
    
    # Create subdirectories
    mkdir -p {logs,data,config,backups,scripts,models}
    
    # Create directory structure documentation
    cat > directory_structure.txt << EOF
Jarvis AI Directory Structure:
├── jarvis_ai/
│   ├── jarvis_agent.py          # Main Jarvis code
│   ├── requirements.txt         # Python dependencies
│   ├── README.md               # Project documentation
│   ├── config/
│   │   ├── .env                 # Environment variables
│   │   └── settings.json        # Configuration settings
│   ├── logs/                    # Application logs
│   ├── data/                    # Database and user data
│   ├── backups/                 # Automatic backups
│   ├── scripts/                 # Utility scripts
│   ├── models/                  # Local AI models (Ollama)
│   └── systemd/                 # System service files
EOF
    
    log "Project structure created in $JARVIS_DIR ✓"
}

# Create requirements.txt file
create_requirements() {
    log "Creating requirements.txt file..."
    
    cat > requirements.txt << 'EOF'
# Core AI and API libraries
openai>=1.3.0
google-generativeai>=0.3.0
aiohttp>=3.8.0
requests>=2.28.0

# Speech recognition and text-to-speech
SpeechRecognition>=3.10.0
pyttsx3>=2.90
pyaudio>=0.2.11

# System monitoring and utilities
psutil>=5.9.0
schedule>=1.2.0

# Additional useful packages for production
python-dotenv>=1.0.0
colorlog>=6.7.0
watchdog>=3.0.0
APScheduler>=3.10.0
cryptography>=41.0.0
httpx>=0.24.0

# Optional: For enhanced audio processing
scipy>=1.11.0
numpy>=1.24.0

# Optional: For better NLP capabilities
nltk>=3.8.0
textblob>=0.17.0

# Development and debugging tools (optional)
black>=23.0.0
pylint>=2.17.0
pytest>=7.4.0
EOF
    
    log "Requirements.txt created ✓"
}

# Setup Python virtual environment
setup_python_env() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv jarvis_env
    
    # Activate virtual environment
    source jarvis_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    log "Python virtual environment created ✓"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    source jarvis_env/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    log "Python dependencies installed ✓"
}

# Setup Ollama (local AI)
setup_ollama() {
    log "Setting up Ollama for local AI processing..."
    
    # Check if Ollama is already installed
    if command -v ollama &> /dev/null; then
        log "Ollama is already installed ✓"
    else
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    # Start Ollama service
    if systemctl is-active --quiet ollama; then
        log "Ollama service is running ✓"
    else
        log_info "Starting Ollama service..."
        sudo systemctl start ollama
        sudo systemctl enable ollama
    fi
    
    # Pull recommended models
    log_info "Pulling recommended AI models (this may take a while)..."
    
    # Pull lightweight model for Raspberry Pi
    ollama pull llama3.2:3b
    
    # Pull larger model if sufficient resources
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    if [[ $TOTAL_MEM -gt 4096 ]]; then
        log_info "Sufficient memory detected, pulling larger model..."
        ollama pull llama3.1:8b
    else
        log_warn "Limited memory - using only lightweight model for now"
    fi
    
    log "Ollama setup completed ✓"
}

# Create configuration files
create_config() {
    log "Creating configuration files..."
    
    # Create .env file
    cat > config/.env << EOF
# API Keys (set these for cloud AI capabilities)
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_MODEL_COMPLEX=llama3.1:8b

# Voice Settings
VOICE_RATE=150
VOICE_VOLUME=0.9

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/jarvis.log

# Security
PRIVACY_MODE=false
LOCAL_ONLY=false
EOF
    
    # Create settings.json
    cat > config/settings.json << EOF
{
    "agent": {
        "name": "Jarvis",
        "version": "2.0",
        "wake_word": "hey jarvis",
        "response_timeout": 30,
        "max_conversation_length": 100
    },
    "voice": {
        "engine": "pyttsx3",
        "rate": 150,
        "volume": 0.9,
        "language": "en-US"
    },
    "ai": {
        "default_provider": "ollama",
        "fallback_provider": "gemini",
        "max_tokens": 1024,
        "temperature": 0.7
    },
    "system": {
        "auto_backup": true,
        "backup_interval_hours": 24,
        "max_log_size_mb": 100,
        "enable_monitoring": true
    }
}
EOF
    
    log "Configuration files created ✓"
}

# Create systemd service
create_systemd_service() {
    log "Creating systemd service for Jarvis..."
    
    # Create systemd service directory
    mkdir -p systemd
    
    # Create service file
    cat > systemd/jarvis.service << EOF
[Unit]
Description=Jarvis AI Assistant
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$HOME/jarvis_ai
Environment=PATH=$HOME/jarvis_ai/jarvis_env/bin
ExecStart=$HOME/jarvis_ai/jarvis_env/bin/python jarvis_agent.py
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$HOME/jarvis_ai
ProtectHome=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jarvis

[Install]
WantedBy=multi-user.target
EOF
    
    # Install service
    sudo cp systemd/jarvis.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log "Systemd service created ✓"
}

# Create utility scripts
create_scripts() {
    log "Creating utility scripts..."
    
    # Start script
    cat > scripts/start_jarvis.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source jarvis_env/bin/activate
python jarvis_agent.py
EOF
    
    # Stop script
    cat > scripts/stop_jarvis.sh << 'EOF'
#!/bin/bash
sudo systemctl stop jarvis
echo "Jarvis stopped"
EOF
    
    # Status script
    cat > scripts/status_jarvis.sh << 'EOF'
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
EOF
    
    # Backup script
    cat > scripts/backup_jarvis.sh << 'EOF'
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
EOF
    
    # Update script
    cat > scripts/update_jarvis.sh << 'EOF'
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
EOF
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    log "Utility scripts created ✓"
}

# Setup security
setup_security() {
    log "Setting up security measures..."
    
    # Configure UFW firewall
    if command -v ufw &> /dev/null; then
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 11434  # Ollama port
        log "Firewall configured ✓"
    fi
    
    # Configure fail2ban
    if command -v fail2ban-client &> /dev/null; then
        sudo systemctl enable fail2ban
        sudo systemctl start fail2ban
        log "Fail2ban configured ✓"
    fi
    
    # Set proper file permissions
    chmod 600 config/.env
    chmod 700 data
    chmod 700 logs
    
    log "Security measures applied ✓"
}

# Post-installation setup
post_install() {
    log "Performing post-installation setup..."
    
    # Create desktop shortcut (if GUI is available)
    if [[ -n "$DISPLAY" ]] && [[ -d "$HOME/Desktop" ]]; then
        cat > "$HOME/Desktop/Jarvis.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Jarvis AI Assistant
Comment=Start Jarvis AI Assistant
Exec=$HOME/jarvis_ai/scripts/start_jarvis.sh
Icon=computer
Terminal=true
Categories=Utility;
EOF
        chmod +x "$HOME/Desktop/Jarvis.desktop"
        log "Desktop shortcut created ✓"
    fi
    
    # Add to PATH (optional)
    if ! grep -q "jarvis_ai/scripts" "$HOME/.bashrc"; then
        echo "export PATH=\$PATH:$HOME/jarvis_ai/scripts" >> "$HOME/.bashrc"
        log "Added Jarvis scripts to PATH ✓"
    fi
    
    # Create initial backup
    ./scripts/backup_jarvis.sh
    
    log "Post-installation setup completed ✓"
}

# Print final instructions
print_instructions() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    INSTALLATION COMPLETE!                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${GREEN}Jarvis AI Assistant has been successfully installed!${NC}"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Configure your API keys in: $HOME/jarvis_ai/config/.env"
    echo "2. Test the installation: $HOME/jarvis_ai/scripts/start_jarvis.sh"
    echo "3. Enable auto-start: sudo systemctl enable jarvis"
    echo
    echo -e "${YELLOW}Available Commands:${NC}"
    echo "• Start Jarvis:    ./scripts/start_jarvis.sh"
    echo "• Stop Jarvis:     ./scripts/stop_jarvis.sh"
    echo "• Check Status:    ./scripts/status_jarvis.sh"
    echo "• Create Backup:   ./scripts/backup_jarvis.sh"
    echo "• Update Jarvis:   ./scripts/update_jarvis.sh"
    echo
    echo -e "${YELLOW}System Service:${NC}"
    echo "• Start service:   sudo systemctl start jarvis"
    echo "• Stop service:    sudo systemctl stop jarvis"
    echo "• Enable auto-start: sudo systemctl enable jarvis"
    echo "• View logs:       journalctl -u jarvis -f"
    echo
    echo -e "${BLUE}Configuration Files:${NC}"
    echo "• Environment:     $HOME/jarvis_ai/config/.env"
    echo "• Settings:        $HOME/jarvis_ai/config/settings.json"
    echo "• Logs:           $HOME/jarvis_ai/logs/"
    echo
    echo -e "${RED}Important Security Notes:${NC}"
    echo "• Never share your API keys"
    echo "• Regularly update the system and Jarvis"
    echo "• Monitor the logs for suspicious activity"
    echo "• Use privacy mode for sensitive conversations"
    echo
    echo -e "${CYAN}Say 'Hey Jarvis' to wake up your assistant!${NC}"
    echo
}

# Main installation function
main() {
    print_banner
    
    log "Starting Jarvis AI Assistant installation..."
    
    check_root
    check_system
    install_system_deps
    test_audio
    create_project_structure
    create_requirements  # Added this line - creates requirements.txt BEFORE trying to use it
    setup_python_env
    install_python_deps
    setup_ollama
    create_config
    create_systemd_service
    create_scripts
    setup_security
    post_install
    
    print_instructions
    
    log "Installation completed successfully! 🎉"
}

# Run main function
main "$@"
