"""
Configuration management utilities
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager for CrowdVision"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        # Load YAML config
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
            print(f"Warning: Config file not found at {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support
        
        Args:
            key: Configuration key (e.g., 'models.detection.type')
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
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_type: Type of model (detection, density, forecasting, etc.)
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'models.{model_type}', {})
    
    def get_training_config(self, model_type: str = None) -> Dict[str, Any]:
        """
        Get training configuration
        
        Args:
            model_type: Optional specific model type
            
        Returns:
            Training configuration dictionary
        """
        if model_type:
            return self.get(f'training.{model_type}', {})
        return self.get('training', {})
    
    @property
    def device(self) -> str:
        """Get compute device (cuda/cpu)"""
        return self.get_env('DEVICE', 'cuda')
    
    @property
    def mlflow_uri(self) -> str:
        """Get MLflow tracking URI"""
        return self.get_env('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        return self.get_env('OPENAI_API_KEY', '')


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
