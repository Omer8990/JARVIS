# 🤖 Jarvis AI Assistant

A powerful, production-ready AI assistant designed for Raspberry Pi and Linux systems. Jarvis combines local AI processing with cloud capabilities to provide a versatile, privacy-conscious voice assistant.

![Jarvis AI](https://img.shields.io/badge/Jarvis-AI%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### 🧠 **Hybrid AI Processing**
- **Local AI**: Ollama integration with LLaMA models for privacy
- **Cloud AI**: OpenAI GPT and Google Gemini support
- **Smart Fallback**: Automatically switches between providers
- **Offline Capable**: Works without internet when using local models

### 🎤 **Advanced Voice Interface**
- Wake word detection ("Hey Jarvis")
- Natural speech recognition
- High-quality text-to-speech
- Multi-language support
- Audio device auto-detection

### 🔧 **System Integration**
- Systemd service for automatic startup
- Comprehensive logging and monitoring
- Automatic backups
- Resource monitoring
- Security hardening

### 🔒 **Privacy & Security**
- Local processing mode
- Encrypted configuration
- Firewall integration
- Fail2ban protection
- Privacy mode for sensitive conversations

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 3B+ or newer (or compatible Linux system)
- Python 3.8+
- 2GB+ RAM recommended
- 4GB+ disk space
- Internet connection for initial setup

### Installation

1. **Download and run the setup script:**
```bash
curl -fsSL https://raw.githubusercontent.com/your-repo/jarvis-ai/main/setup_jarvis.sh | bash
```

2. **Or clone and run manually:**
```bash
git clone https://github.com/your-repo/jarvis-ai.git
cd jarvis-ai
chmod +x setup_jarvis.sh
./setup_jarvis.sh
```

3. **Configure API keys** (optional for cloud features):
```bash
nano ~/jarvis_ai/config/.env
```

4. **Start Jarvis:**
```bash
cd ~/jarvis_ai
./scripts/start_jarvis.sh
```

## 📁 Project Structure

```
jarvis_ai/
├── jarvis_agent.py          # Main application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── setup_jarvis.sh         # Installation script
├── config/
│   ├── .env                # Environment variables (API keys)
│   └── settings.json       # Configuration settings
├── logs/                   # Application logs
├── data/                   # Database and user data
├── backups/               # Automatic backups
├── scripts/               # Utility scripts
│   ├── start_jarvis.sh    # Start Jarvis
│   ├── stop_jarvis.sh     # Stop Jarvis
│   ├── status_jarvis.sh   # Check status
│   ├── backup_jarvis.sh   # Create backup
│   └── update_jarvis.sh   # Update system
├── models/                # Local AI models
└── systemd/              # System service files
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# API Keys (optional - for cloud AI)
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Voice Settings
VOICE_RATE=150
VOICE_VOLUME=0.9

# Privacy Settings
PRIVACY_MODE=false
LOCAL_ONLY=false
```

### Settings (settings.json)
```json
{
    "agent": {
        "name": "Jarvis",
        "wake_word": "hey jarvis",
        "response_timeout": 30
    },
    "voice": {
        "engine": "pyttsx3",
        "rate": 150,
        "volume": 0.9
    },
    "ai": {
        "default_provider": "ollama",
        "fallback_provider": "gemini"
    }
}
```

## 🎯 Usage

### Voice Commands
- **"Hey Jarvis"** - Wake up the assistant
- **"What's the weather?"** - Get weather information
- **"Set a timer for 5 minutes"** - Set timers and reminders
- **"What's the system status?"** - Check system resources
- **"Privacy mode on/off"** - Toggle privacy mode

### Command Line Interface
```bash
# Start Jarvis
./scripts/start_jarvis.sh

# Check status
./scripts/status_jarvis.sh

# Create backup
./scripts/backup_jarvis.sh

# Update system
./scripts/update_jarvis.sh
```

### System Service
```bash
# Enable auto-start
sudo systemctl enable jarvis

# Start/stop service
sudo systemctl start jarvis
sudo systemctl stop jarvis

# View logs
journalctl -u jarvis -f
```

## 🔧 Advanced Configuration

### Adding Custom Commands
Extend Jarvis by modifying the `jarvis_agent.py` file:

```python
def handle_custom_command(self, command):
    if "custom action" in command.lower():
        # Your custom logic here
        return "Custom response"
```

### Multiple AI Providers
Configure fallback chains in `settings.json`:

```json
{
    "ai": {
        "providers": ["ollama", "openai", "gemini"],
        "fallback_enabled": true
    }
}
```

### Hardware Integration
Jarvis can integrate with:
- GPIO pins for hardware control
- Camera modules for vision
- Sensors for environmental monitoring
- LED strips for visual feedback

## 🛠️ Troubleshooting

### Common Issues

**Audio not working:**
```bash
# Check audio devices
python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'Device {i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]; p.terminate()"

# Test speakers
espeak "Test audio"
```

**Ollama not responding:**
```bash
# Check Ollama status
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Pull models again
ollama pull llama3.2:3b
```

**Permission errors:**
```bash
# Fix permissions
chmod +x scripts/*.sh
chmod 600 config/.env
```

### Log Analysis
```bash
# View Jarvis logs
tail -f ~/jarvis_ai/logs/jarvis.log

# View system logs
journalctl -u jarvis -f

# Check Ollama logs
journalctl -u ollama -f
```

## 🔐 Security

### Best Practices
- Keep API keys secure and never commit them to version control
- Use local-only mode for sensitive conversations
- Regularly update the system and dependencies
- Monitor logs for suspicious activity
- Use strong passwords and SSH keys

### Security Features
- Automatic firewall configuration
- Fail2ban integration
- Encrypted configuration files
- Process isolation
- Network access restrictions

## 🚀 Performance Optimization

### Raspberry Pi Optimization
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Enable hardware acceleration
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt

# Optimize for audio
echo 'audio_pwm_mode=2' | sudo tee -a /boot/config.txt
```

### Model Selection
- **Raspberry Pi 3B+**: llama3.2:3b (lightweight)
- **Raspberry Pi 4 4GB+**: llama3.1:8b (better performance)
- **High-end systems**: llama3.1:70b (best quality)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-repo/jarvis-ai.git
cd jarvis-ai

# Create development environment
python3 -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt

# Install development tools
pip install black pylint pytest
```

### Testing
```bash
# Run tests
pytest tests/

# Code formatting
black jarvis_agent.py

# Linting
pylint jarvis_agent.py
```

## 📊 Monitoring

### System Metrics
- CPU and memory usage
- Disk space monitoring
- Network connectivity
- AI model performance
- Response times

### Health Checks
```bash
# System health
./scripts/status_jarvis.sh

# Detailed monitoring
htop
iotop
nethogs
```

## 🔄 Updates

###