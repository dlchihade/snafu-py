"""
Configuration management for semantic fluency analysis
"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters"""
    
    # Analysis parameters
    semantic_weight: float = 0.7
    clustering_threshold: float = 0.5
    similarity_threshold: float = 0.6  # Threshold for exploitation vs exploration phases
    cache_size: int = 1000
    
    # spaCy Model Configuration
    spacy_model: str = "en_core_web_md"
    batch_size: int = 1000
    min_similarity: float = 0.1
    max_similarity: float = 0.95
    
    # Phase Analysis Parameters
    min_phase_length: int = 2
    max_phase_length: int = 20
    
    # Performance Settings
    enable_caching: bool = True
    enable_batch_processing: bool = True
    enable_parallel_processing: bool = False
    
    # Output settings
    output_dir: str = "output"
    save_plots: bool = True
    plot_format: str = "svg"
    
    # Data paths
    data_paths: Dict[str, str] = None
    
    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = {
                'fluency_data': 'data/fluency_data.csv',
                'meg_data': 'data/meg_data.csv'
            }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Handle nested data_paths
            if 'data_paths' in config_dict:
                data_paths = config_dict.pop('data_paths')
            else:
                data_paths = None
            
            config = cls(**config_dict)
            if data_paths:
                config.data_paths = data_paths
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            config_dict = {
                'semantic_weight': self.semantic_weight,
                'clustering_threshold': self.clustering_threshold,
                'similarity_threshold': self.similarity_threshold,
                'cache_size': self.cache_size,
                'spacy_model': self.spacy_model,
                'batch_size': self.batch_size,
                'min_similarity': self.min_similarity,
                'max_similarity': self.max_similarity,
                'min_phase_length': self.min_phase_length,
                'max_phase_length': self.max_phase_length,
                'enable_caching': self.enable_caching,
                'enable_batch_processing': self.enable_batch_processing,
                'enable_parallel_processing': self.enable_parallel_processing,
                'output_dir': self.output_dir,
                'save_plots': self.save_plots,
                'plot_format': self.plot_format,
                'data_paths': self.data_paths
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Analysis parameters
        if not (0 <= self.semantic_weight <= 1):
            errors.append("semantic_weight must be between 0 and 1")
        
        if not (0 <= self.clustering_threshold <= 1):
            errors.append("clustering_threshold must be between 0 and 1")
        
        if not (0 <= self.similarity_threshold <= 1):
            errors.append("similarity_threshold must be between 0 and 1")
        
        if self.cache_size < 1:
            errors.append("cache_size must be positive")
        
        # spaCy parameters
        if not (0 <= self.min_similarity <= 1):
            errors.append("min_similarity must be between 0 and 1")
        
        if not (0 <= self.max_similarity <= 1):
            errors.append("max_similarity must be between 0 and 1")
        
        if self.min_similarity >= self.max_similarity:
            errors.append("min_similarity must be less than max_similarity")
        
        if self.batch_size < 1:
            errors.append("batch_size must be positive")
        
        # Phase parameters
        if self.min_phase_length < 1:
            errors.append("min_phase_length must be positive")
        
        if self.max_phase_length < self.min_phase_length:
            errors.append("max_phase_length must be >= min_phase_length")
        
        # Output parameters
        if self.plot_format not in ['svg', 'png', 'pdf', 'jpg']:
            errors.append("plot_format must be one of: svg, png, pdf, jpg")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
