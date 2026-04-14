"""
Configuration Module
Load and manage application configuration
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config_path = Path(config_file)
        self.config = self._load_config()
        
        # Application settings
        self.app_name = self.get("app.name", "AI Chatbot")
        self.app_version = self.get("app.version", "1.0.0")
        self.debug = self.get("app.debug", False)
        
        # Backend settings
        self.host = self.get("backend.host", "localhost")
        self.port = self.get("backend.port", 5000)
        self.workers = self.get("backend.workers", 4)
        self.timeout = self.get("backend.timeout", 30)
        
        # NLP settings
        self.nlp_model = self.get("nlp.model", "distilbert-base-uncased")
        self.max_tokens = self.get("nlp.max_tokens", 512)
        self.temperature = self.get("nlp.temperature", 0.7)
        self.top_p = self.get("nlp.top_p", 0.9)
        
        # Database settings
        self.database_path = self.get("database.path", "data/conversations.db")
        self.auto_backup = self.get("database.auto_backup", True)
        
        # Web settings
        self.web_port = self.get("web.port", 8080)
        self.web_debug = self.get("web.debug", False)
        self.session_timeout = self.get("web.session_timeout", 3600)
        
        # CLI settings
        self.cli_history_file = self.get("cli.history_file", "data/cli_history.txt")
        self.cli_max_history = self.get("cli.max_history", 1000)
        
        # System settings
        self.log_level = self.get("system.log_level", "INFO")
        self.log_file = self.get("system.log_file", "logs/app.log")
        self.max_log_size = self.get("system.max_log_size", 10485760)
        
        logger.info(f"Configuration loaded from {config_file}")
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., "database.path")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> dict:
        """Get all configuration as dictionary"""
        return self.config.copy()
    
    def save(self, filepath: str = None) -> bool:
        """
        Save configuration to file
        
        Args:
            filepath: Optional filepath (uses default if not provided)
            
        Returns:
            Success status
        """
        try:
            path = Path(filepath or self.config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = ["app.name", "backend.host", "nlp.model", "database.path"]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        return f"Config({self.app_name} v{self.app_version})"
