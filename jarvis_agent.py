async def _show_api_usage(self):
    """Show API usage statistics"""
    ollama_usage = self.db.get_daily_api_usage('ollama')
    gemini_usage = self.db.get_daily_api_usage('gemini')
    openai_usage = self.db.get_daily_api_usage('openai')
    
    status = f"Today's API usage: Ollama {ollama_usage['requests']} requests, "
    status += f"Gemini {gemini_usage['requests']} requests, "
    status += f"OpenAI {openai_usage['requests']} requests. "
    
    if self.api_manager.ollama_available:
        status += "Ollama is available for private processing."
    else:
        status += "Ollama is not available."
    
    self.voice.speak(status)

async def _force_ollama_mode(self):
    """Force using Ollama for next interactions"""
    if self.api_manager.ollama_available:
        self.privacy_mode = True
        self.voice.speak("Switching to Ollama mode for private processing.")
    else:
        self.voice.speak("Ollama is not available. Please check if it's running.")

async def _enable_privacy_mode(self):
    """Enable privacy mode (Ollama only)"""
    if self.api_manager.ollama_available:
        self.privacy_mode = True
        self.voice.speak("Privacy mode enabled. All processing will be done locally with Ollama.")
    else:
        self.voice.speak("Privacy mode requires Ollama. Please install and start Ollama first.")

# Command handling logic - put this inside your main command processing method
async def process_command(self, input_text):
    """Process voice commands"""
    input_lower = input_text.lower()
    
    if 'api status' in input_lower or 'usage status' in input_lower:
        await self._show_api_usage()
        return True
    elif 'switch to ollama' in input_lower:
        await self._force_ollama_mode()
        return True
    elif 'privacy mode' in input_lower:
        await self._enable_privacy_mode()
        return True
    
    # Add your other command handling logic here
    return False
# Core system with memory, task execution, and hybrid API usage

import asyncio
import json
import sqlite3
import logging
import os
import sys
import time
import threading
import queue
import speech_recognition as sr
import pyttsx3
import requests
import openai
import google.generativeai as genai
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import subprocess
import schedule
import pickle
from pathlib import Path

# Configuration
class Config:
    # API Keys (set these as environment variables)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')  # Lightweight model for Pi 5
    OLLAMA_MODEL_COMPLEX = os.getenv('OLLAMA_MODEL_COMPLEX', 'llama3.1:8b')  # For complex tasks
    
    # Database
    DB_PATH = 'jarvis_memory.db'
    
    # Voice settings
    VOICE_RATE = 150
    VOICE_VOLUME = 0.9
    
    # API Usage limits (daily)
    GEMINI_DAILY_LIMIT = 1500  # Free tier limit
    OPENAI_DAILY_LIMIT = 100   # Conservative limit to minimize cost
    OLLAMA_DAILY_LIMIT = 10000  # Local, so high limit
    
    # Complexity thresholds
    SIMPLE_TASK_KEYWORDS = [
        'hello', 'hi', 'time', 'date', 'weather', 'system', 'status', 
        'remember', 'task', 'remind', 'calculate', 'convert', 'define'
    ]
    COMPLEX_TASK_KEYWORDS = [
        'analyze', 'write', 'create', 'generate', 'explain complex', 
        'research', 'compare', 'summarize long', 'translate', 'code review'
    ]
    
    # System paths
    BASE_DIR = Path(__file__).parent
    LOGS_DIR = BASE_DIR / 'logs'
    DATA_DIR = BASE_DIR / 'data'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'jarvis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('Jarvis')

# Data structures
@dataclass
class Memory:
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    memory_type: str = 'conversation'  # conversation, task, fact, preference
    content: str = ''
    importance: int = 5  # 1-10 scale
    tags: List[str] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.context is None:
            self.context = {}

@dataclass
class Task:
    id: Optional[int] = None
    title: str = ''
    description: str = ''
    status: str = 'pending'  # pending, in_progress, completed, failed
    priority: int = 5  # 1-10 scale
    created_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.context is None:
            self.context = {}

class APIProvider(Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"
    BASIC = "basic"  # Fallback for basic responses

# Database Manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    tags TEXT,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    due_date TEXT,
                    completed_at TEXT,
                    context TEXT
                )
            ''')
            
            # API usage tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    date TEXT NOT NULL,
                    requests_count INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    model_used TEXT
                )
            ''')
            
            # User preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def store_memory(self, memory: Memory) -> int:
        """Store a memory and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories (timestamp, memory_type, content, importance, tags, context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                memory.timestamp.isoformat(),
                memory.memory_type,
                memory.content,
                memory.importance,
                json.dumps(memory.tags),
                json.dumps(memory.context)
            ))
            return cursor.lastrowid

    def get_memories(self, memory_type: str = None, limit: int = 100) -> List[Memory]:
        """Retrieve memories, optionally filtered by type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if memory_type:
                cursor.execute('''
                    SELECT * FROM memories WHERE memory_type = ? 
                    ORDER BY importance DESC, timestamp DESC LIMIT ?
                ''', (memory_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM memories ORDER BY importance DESC, timestamp DESC LIMIT ?
                ''', (limit,))
            
            memories = []
            for row in cursor.fetchall():
                memory = Memory(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    memory_type=row[2],
                    content=row[3],
                    importance=row[4],
                    tags=json.loads(row[5]) if row[5] else [],
                    context=json.loads(row[6]) if row[6] else {}
                )
                memories.append(memory)
            
            return memories

    def store_task(self, task: Task) -> int:
        """Store a task and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (title, description, status, priority, created_at, due_date, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.title,
                task.description,
                task.status,
                task.priority,
                task.created_at.isoformat(),
                task.due_date.isoformat() if task.due_date else None,
                json.dumps(task.context)
            ))
            return cursor.lastrowid

    def update_task_status(self, task_id: int, status: str):
        """Update task status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            completed_at = datetime.now().isoformat() if status == 'completed' else None
            cursor.execute('''
                UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?
            ''', (status, completed_at, task_id))

    def get_tasks(self, status: str = None) -> List[Task]:
        """Retrieve tasks, optionally filtered by status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute('SELECT * FROM tasks WHERE status = ? ORDER BY priority DESC, created_at DESC', (status,))
            else:
                cursor.execute('SELECT * FROM tasks ORDER BY priority DESC, created_at DESC')
            
            tasks = []
            for row in cursor.fetchall():
                task = Task(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    status=row[3],
                    priority=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    due_date=datetime.fromisoformat(row[6]) if row[6] else None,
                    completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    context=json.loads(row[8]) if row[8] else {}
                )
                tasks.append(task)
            
            return tasks

    def track_api_usage(self, provider: str, requests: int = 1, tokens: int = 0, cost: float = 0.0, model: str = ''):
        """Track API usage"""
        today = datetime.now().date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO api_usage (provider, date, requests_count, tokens_used, cost, model_used)
                VALUES (?, ?, 
                    COALESCE((SELECT requests_count FROM api_usage WHERE provider = ? AND date = ?), 0) + ?,
                    COALESCE((SELECT tokens_used FROM api_usage WHERE provider = ? AND date = ?), 0) + ?,
                    COALESCE((SELECT cost FROM api_usage WHERE provider = ? AND date = ?), 0) + ?,
                    ?
                )
            ''', (provider, today, provider, today, requests, provider, today, tokens, provider, today, cost, model))

    def get_daily_api_usage(self, provider: str) -> Dict[str, int]:
        """Get today's API usage for a provider"""
        today = datetime.now().date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT requests_count, tokens_used, cost FROM api_usage 
                WHERE provider = ? AND date = ?
            ''', (provider, today))
            
            result = cursor.fetchone()
            return {
                'requests': result[0] if result else 0,
                'tokens': result[1] if result else 0,
                'cost': result[2] if result else 0.0
            }

# API Manager
class APIManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
        # Initialize APIs
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
        
        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
        # Test Ollama connection
        self.ollama_available = self._test_ollama_connection()
        if self.ollama_available:
            logger.info(f"Ollama connected successfully at {Config.OLLAMA_HOST}")
        else:
            logger.warning("Ollama not available - falling back to cloud APIs")

    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{Config.OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _classify_task_complexity(self, prompt: str) -> str:
        """Classify task complexity based on content"""
        prompt_lower = prompt.lower()
        
        # Check for complex task indicators
        complex_indicators = sum(1 for keyword in Config.COMPLEX_TASK_KEYWORDS if keyword in prompt_lower)
        simple_indicators = sum(1 for keyword in Config.SIMPLE_TASK_KEYWORDS if keyword in prompt_lower)
        
        # Length-based complexity
        word_count = len(prompt.split())
        
        if complex_indicators > 0 or word_count > 50:
            return 'complex'
        elif simple_indicators > 0 or word_count < 15:
            return 'simple'
        else:
            return 'medium'

    def choose_provider(self, complexity: str = 'simple', privacy_required: bool = False) -> APIProvider:
        """Choose the best API provider based on usage, complexity, and privacy needs"""
        ollama_usage = self.db.get_daily_api_usage('ollama')
        gemini_usage = self.db.get_daily_api_usage('gemini')
        openai_usage = self.db.get_daily_api_usage('openai')
        
        # For private/personal tasks, prefer Ollama
        if privacy_required and self.ollama_available:
            return APIProvider.OLLAMA
        
        # For simple tasks, prefer Ollama (free and private)
        if complexity == 'simple':
            if self.ollama_available and ollama_usage['requests'] < Config.OLLAMA_DAILY_LIMIT:
                return APIProvider.OLLAMA
            elif gemini_usage['requests'] < Config.GEMINI_DAILY_LIMIT:
                return APIProvider.GEMINI
            else:
                return APIProvider.BASIC
        
        # For medium complexity, try Ollama first, then Gemini
        elif complexity == 'medium':
            if self.ollama_available and ollama_usage['requests'] < Config.OLLAMA_DAILY_LIMIT:
                return APIProvider.OLLAMA
            elif gemini_usage['requests'] < Config.GEMINI_DAILY_LIMIT:
                return APIProvider.GEMINI
            elif openai_usage['requests'] < Config.OPENAI_DAILY_LIMIT and Config.OPENAI_API_KEY:
                return APIProvider.OPENAI
            else:
                return APIProvider.BASIC
        
        # For complex tasks, prefer cloud APIs
        else:  # complex
            if gemini_usage['requests'] < Config.GEMINI_DAILY_LIMIT:
                return APIProvider.GEMINI
            elif openai_usage['requests'] < Config.OPENAI_DAILY_LIMIT and Config.OPENAI_API_KEY:
                return APIProvider.OPENAI
            elif self.ollama_available and ollama_usage['requests'] < Config.OLLAMA_DAILY_LIMIT:
                return APIProvider.OLLAMA
            else:
                return APIProvider.BASIC

    async def generate_response(self, prompt: str, context: str = "", privacy_required: bool = False) -> str:
        """Generate a response using the best available provider"""
        complexity = self._classify_task_complexity(prompt)
        provider = self.choose_provider(complexity, privacy_required)
        
        logger.info(f"Using {provider.value} for {complexity} task")
        
        try:
            if provider == APIProvider.OLLAMA:
                return await self._generate_ollama(prompt, context, complexity)
            elif provider == APIProvider.GEMINI:
                return await self._generate_gemini(prompt, context)
            elif provider == APIProvider.OPENAI:
                return await self._generate_openai(prompt, context)
            else:
                return await self._generate_basic(prompt, context)
        except Exception as e:
            logger.error(f"Error generating response with {provider.value}: {e}")
            # Try fallback
            if provider != APIProvider.BASIC:
                return await self._generate_basic(prompt, context)
            return "I'm having trouble processing that request right now. Please try again."

    async def _generate_ollama(self, prompt: str, context: str = "", complexity: str = 'simple') -> str:
        """Generate response using Ollama"""
        # Choose model based on complexity
        model = Config.OLLAMA_MODEL_COMPLEX if complexity == 'complex' else Config.OLLAMA_MODEL
        
        # Build the full prompt with context
        system_prompt = f"""You are Jarvis, a helpful AI assistant running locally on a Raspberry Pi. 
You are private, secure, and always prioritize user privacy. Be concise but helpful.
{context}"""
        
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 4096,
                "num_predict": 512 if complexity == 'simple' else 1024
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{Config.OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.db.track_api_usage('ollama', requests=1, model=model)
                        return result.get('response', 'No response generated')
                    else:
                        raise Exception(f"Ollama API returned status {response.status}")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def _generate_gemini(self, prompt: str, context: str = "") -> str:
        """Generate response using Gemini API"""
        full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
        
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, full_prompt
            )
            self.db.track_api_usage('gemini', requests=1)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def _generate_openai(self, prompt: str, context: str = "") -> str:
        """Generate response using OpenAI API"""
        messages = [
            {"role": "system", "content": context or "You are Jarvis, a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            self.db.track_api_usage('openai', requests=1, tokens=response.usage.total_tokens)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _generate_basic(self, prompt: str, context: str = "") -> str:
        """Generate simple response using basic pattern matching"""
        # Basic pattern matching and responses
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif any(word in prompt_lower for word in ['time', 'clock']):
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif any(word in prompt_lower for word in ['date', 'today']):
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
        elif any(word in prompt_lower for word in ['weather']):
            return "I'd need internet access to check the weather. You can ask me to set up weather monitoring."
        elif any(word in prompt_lower for word in ['system', 'cpu', 'memory']):
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            temp = self._get_cpu_temperature()
            return f"System status - CPU: {cpu_percent}%, Memory: {memory.percent}% used, CPU temp: {temp}°C"
        elif any(word in prompt_lower for word in ['shutdown', 'restart']):
            return "I cannot perform system shutdown or restart commands without explicit confirmation for safety reasons."
        elif 'ollama' in prompt_lower:
            if self.ollama_available:
                return f"Ollama is running with model {Config.OLLAMA_MODEL}. I'm using it for private, local AI processing."
            else:
                return "Ollama is not currently available. Please check if it's running."
        else:
            return "I understand you're asking about something, but I'm operating in basic mode with limited capabilities. Could you rephrase your request or ask me to remember this for when I have full AI access?"

    def _get_cpu_temperature(self) -> str:
        """Get CPU temperature on Raspberry Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000
                return f"{temp:.1f}"
        except:
            return "N/A"

# Voice Interface
class VoiceInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS
        self.tts_engine.setProperty('rate', Config.VOICE_RATE)
        self.tts_engine.setProperty('volume', Config.VOICE_VOLUME)
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for voice input"""
        try:
            with self.microphone as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Heard: {text}")
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None

    def speak(self, text: str):
        """Convert text to speech"""
        logger.info(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

# Task Executor
class TaskExecutor:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.running_tasks = {}

    async def execute_task(self, task: Task) -> bool:
        """Execute a task based on its description"""
        try:
            self.db.update_task_status(task.id, 'in_progress')
            success = False
            
            # Parse task type and execute accordingly
            task_lower = task.description.lower()
            
            if 'reminder' in task_lower or 'remind' in task_lower:
                success = await self._handle_reminder(task)
            elif 'system' in task_lower:
                success = await self._handle_system_task(task)
            elif 'web' in task_lower or 'search' in task_lower:
                success = await self._handle_web_task(task)
            elif 'file' in task_lower:
                success = await self._handle_file_task(task)
            else:
                success = await self._handle_general_task(task)
            
            status = 'completed' if success else 'failed'
            self.db.update_task_status(task.id, status)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            self.db.update_task_status(task.id, 'failed')
            return False

    async def _handle_reminder(self, task: Task) -> bool:
        """Handle reminder tasks"""
        # This would integrate with system notifications
        logger.info(f"Setting reminder: {task.title}")
        return True

    async def _handle_system_task(self, task: Task) -> bool:
        """Handle system-related tasks"""
        try:
            if 'restart' in task.description.lower():
                logger.warning("System restart requested - this should be confirmed by user")
                return False  # Safety measure
            elif 'update' in task.description.lower():
                result = subprocess.run(['sudo', 'apt', 'update'], capture_output=True, text=True)
                return result.returncode == 0
            else:
                return True
        except Exception as e:
            logger.error(f"System task error: {e}")
            return False

    async def _handle_web_task(self, task: Task) -> bool:
        """Handle web-related tasks"""
        # Placeholder for web scraping, API calls, etc.
        logger.info(f"Web task: {task.description}")
        return True

    async def _handle_file_task(self, task: Task) -> bool:
        """Handle file operations"""
        logger.info(f"File task: {task.description}")
        return True

    async def _handle_general_task(self, task: Task) -> bool:
        """Handle general tasks"""
        logger.info(f"General task: {task.description}")
        return True

# Memory System
class MemorySystem:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.short_term_memory = []
        self.context_window = 10

    def add_memory(self, content: str, memory_type: str = 'conversation', 
                   importance: int = 5, tags: List[str] = None, context: Dict = None):
        """Add a new memory"""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            context=context or {}
        )
        
        # Store in database
        memory_id = self.db.store_memory(memory)
        memory.id = memory_id
        
        # Add to short-term memory
        self.short_term_memory.append(memory)
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > self.context_window:
            self.short_term_memory.pop(0)
        
        logger.info(f"Stored memory: {content[:50]}...")

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Memory]:
        """Get memories relevant to a query"""
        # Simple keyword matching - could be enhanced with embeddings
        query_words = set(query.lower().split())
        
        all_memories = self.db.get_memories(limit=100)
        scored_memories = []
        
        for memory in all_memories:
            content_words = set(memory.content.lower().split())
            tag_words = set(' '.join(memory.tags).lower().split())
            
            # Calculate relevance score
            content_overlap = len(query_words.intersection(content_words))
            tag_overlap = len(query_words.intersection(tag_words))
            
            score = content_overlap + (tag_overlap * 2) + (memory.importance / 10)
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(reverse=True)
        return [memory for score, memory in scored_memories[:limit]]

    def get_context_for_conversation(self) -> str:
        """Get context string for AI conversation"""
        context_parts = []
        
        # Add recent memories
        recent_memories = self.short_term_memory[-5:]
        if recent_memories:
            context_parts.append("Recent conversation:")
            for memory in recent_memories:
                context_parts.append(f"- {memory.content}")
        
        # Add important facts
        important_memories = self.db.get_memories(memory_type='fact', limit=3)
        if important_memories:
            context_parts.append("\nImportant facts to remember:")
            for memory in important_memories:
                context_parts.append(f"- {memory.content}")
        
        return "\n".join(context_parts)

# Main Jarvis Agent
class JarvisAgent:
    def __init__(self):
        # Create necessary directories
        Config.LOGS_DIR.mkdir(exist_ok=True)
        Config.DATA_DIR.mkdir(exist_ok=True)
        
        # Initialize components
        self.db = DatabaseManager(Config.DB_PATH)
        self.api_manager = APIManager(self.db)
        self.voice = VoiceInterface()
        self.memory = MemorySystem(self.db)
        self.task_executor = TaskExecutor(self.db)
        
        # State
        self.is_listening = False
        self.is_running = True
        self.privacy_mode = False
        self.command_queue = queue.Queue()
        
        # Load user preferences
        self.load_preferences()
        
        logger.info("Jarvis Agent initialized successfully")

    def load_preferences(self):
        """Load user preferences from database"""
        # This would load user-specific settings
        pass

    async def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Store user input in memory
        self.memory.add_memory(f"User said: {user_input}", 'conversation')
        
        # Get context for AI
        context = self.memory.get_context_for_conversation()
        
        # Check for special commands
        if await self._handle_special_commands(user_input):
            return "Command executed."
        
        # Determine if privacy is required (personal info, sensitive topics)
        privacy_required = self._requires_privacy(user_input)
        
        # Generate AI response
        response = await self.api_manager.generate_response(user_input, context, privacy_required)
        
        # Store AI response in memory
        self.memory.add_memory(f"Assistant responded: {response}", 'conversation')
        
        return response

    def _requires_privacy(self, user_input: str) -> bool:
        """Determine if the request requires privacy (local processing)"""
        private_keywords = [
            'personal', 'private', 'secret', 'password', 'confidential',
            'my name', 'my address', 'my phone', 'my email', 'my location',
            'bank', 'credit card', 'ssn', 'social security', 'medical',
            'remember my', 'store my', 'save my'
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in private_keywords)

    async def _handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands that don't need AI processing"""
        input_lower = user_input.lower()
        
        if 'create task' in input_lower or 'add task' in input_lower:
            await self._create_task_from_input(user_input)
            return True
        elif 'show tasks' in input_lower or 'list tasks' in input_lower:
            await self._show_tasks()
            return True
        elif 'system status' in input_lower:
            await self._show_system_status()
            return True
        elif 'remember that' in input_lower or 'remember this' in input_lower:
            await self._store_explicit_memory(user_input)
            return True
        
        return False

    async def _create_task_from_input(self, user_input: str):
        """Create a task from user input"""
        # Extract task details (could be enhanced with NLP)
        title = user_input.replace('create task', '').replace('add task', '').strip()
        
        task = Task(
            title=title,
            description=title,
            priority=5
        )
        
        task_id = self.db.store_task(task)
        self.voice.speak(f"Task created: {title}")
        logger.info(f"Created task {task_id}: {title}")

    async def _show_tasks(self):
        """Show pending tasks"""
        tasks = self.db.get_tasks(status='pending')
        
        if not tasks:
            self.voice.speak("You have no pending tasks.")
            return
        
        response = f"You have {len(tasks)} pending tasks: "
        for i, task in enumerate(tasks[:3], 1):  # Show max 3 tasks
            response += f"{i}. {task.title}. "
        
        self.voice.speak(response)

    async def _show_system_status(self):
        """Show system status"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = f"System status: CPU at {cpu_percent}%, Memory at {memory.percent}%, Disk at {disk.percent}% used."
        self.voice.speak(status)

    async def _store_explicit_memory(self, user_input: str):
        """Store explicit memory from user"""
        content = user_input.replace('remember that', '').replace('remember this', '').strip()
        self.memory.add_memory(content, 'fact', importance=8)
        self.voice.speak("I'll remember that.")

    def start_voice_loop(self):
        """Start the voice interaction loop"""
        def voice_loop():
            while self.is_running:
                try:
                    if self.is_listening:
                        user_input = self.voice.listen(timeout=1)
                        if user_input:
                            self.command_queue.put(user_input)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Voice loop error: {e}")
        
        voice_thread = threading.Thread(target=voice_loop, daemon=True)
        voice_thread.start()

    async def main_loop(self):
        """Main processing loop"""
        logger.info("Jarvis is now active. Say 'Hey Jarvis' to start listening.")
        self.voice.speak("Jarvis is now active. Say Hey Jarvis to start listening.")
        
        while self.is_running:
            try:
                # Process voice commands
                if not self.command_queue.empty():
                    user_input = self.command_queue.get()
                    
                    if 'hey jarvis' in user_input.lower():
                        self.is_listening = True
                        self.voice.speak("Yes, how can I help you?")
                        continue
                    
                    if self.is_listening:
                        if 'stop listening' in user_input.lower() or 'goodbye' in user_input.lower():
                            self.is_listening = False
                            self.voice.speak("I'll stop listening now. Say Hey Jarvis when you need me.")
                            continue
                        
                        # Process the input
                        response = await self.process_input(user_input)
                        self.voice.speak(response)
                
                # Execute pending tasks
                await self._execute_pending_tasks()
                
                # Brief pause
                await asyncio.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("Shutting down Jarvis...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(1)

    async def _execute_pending_tasks(self):
        """Execute pending tasks"""
        tasks = self.db.get_tasks(status='pending')
        
        for task in tasks[:1]:  # Execute one task at a time
            if task.due_date and task.due_date <= datetime.now():
                success = await self.task_executor.execute_task(task)
                if success:
                    logger.info(f"Completed task: {task.title}")
                else:
                    logger.warning(f"Failed to complete task: {task.title}")

    def shutdown(self):
        """Shutdown the agent gracefully"""
        self.is_running = False
        logger.info("Jarvis Agent shutting down...")

# Main execution
async def main():
    # Check for required API keys
    if not Config.GEMINI_API_KEY and not Config.OPENAI_API_KEY:
        print("WARNING: No API keys configured. Agent will run in local-only mode with limited capabilities.")
        print("Set GEMINI_API_KEY and/or OPENAI_API_KEY environment variables for full functionality.")
    
    # Initialize and run Jarvis
    jarvis = JarvisAgent()
    
    try:
        # Start voice processing in background
        jarvis.start_voice_loop()
        
        # Run main loop
        await jarvis.main_loop()
    
    except KeyboardInterrupt:
        pass
    finally:
        jarvis.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
