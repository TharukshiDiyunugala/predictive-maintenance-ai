"""
Configuration Manager for Predictive Maintenance System
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._override_with_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            # Try example config
            example_path = self.config_path.parent / 'config.example.yaml'
            if example_path.exists():
                print(f"Warning: Using example config from {example_path}")
                self.config_path = example_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _override_with_env(self):
        """Override configuration with environment variables"""
        # Database
        if os.getenv('DB_HOST'):
            self.config['database']['host'] = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.config['database']['port'] = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self.config['database']['name'] = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self.config['database']['user'] = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.config['database']['password'] = os.getenv('DB_PASSWORD')
        
        # Redis
        if os.getenv('REDIS_HOST'):
            self.config['redis']['host'] = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            self.config['redis']['port'] = int(os.getenv('REDIS_PORT'))
        
        # API
        if os.getenv('API_HOST'):
            self.config['api']['host'] = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.config['api']['port'] = int(os.getenv('API_PORT'))
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by dot notation key
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        db = self.config['database']
        return f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        redis = self.config['redis']
        password_part = f":{redis['password']}@" if redis.get('password') else ""
        return f"redis://{password_part}{redis['host']}:{redis['port']}/{redis['db']}"


# Global configuration instance
config = Config()
