# ==================== src/config.py ====================
"""
Configuration management for uplift-learn

Centralized configuration loading and validation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """File path configuration"""
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"
    exports_dir: str = "exports"
    plots_dir: str = "plots"
    logs_dir: str = "logs"
    
    rfm_data: str = "data/x5_rfm_processed.pkl"
    predictions: str = "results/tlearner_predictions.csv"
    tlearner_model: str = "models/tlearner_model.pkl"
    calibrator: str = "models/calibrator.pkl"
    calibrated_predictions: str = "results/final_cate.csv"
    
    def ensure_dirs(self):
        """Create directories if they don't exist"""
        for dir_path in [
            self.data_dir, self.results_dir, self.models_dir,
            self.exports_dir, self.plots_dir, self.logs_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class CampaignConfig:
    """Campaign parameters"""
    margin: float = 50.0
    contact_cost: float = 0.50
    budget: float = 10000.0
    test_split_ratio: float = 0.20
    alpha: float = 0.05
    power: float = 0.80
    min_roi: float = 0.0
    top_k_default: float = 0.30


@dataclass
class ModelConfig:
    """Model parameters"""
    random_state: int = 42
    test_size: float = 0.20
    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'verbosity': 0
    })


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    qini_bins: int = 100
    uplift_k_values: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    confidence_level: float = 0.95


@dataclass
class CalibrationConfig:
    """Calibration settings"""
    method: str = "isotonic"
    cv_folds: int = 5
    min_samples_leaf: int = 10


@dataclass
class PlottingConfig:
    """Plotting configuration"""
    dpi: int = 150
    figsize: tuple = (10, 6)
    style: str = "seaborn-v0_8-darkgrid"
    color_palette: list = field(default_factory=lambda: [
        "#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#D62828"
    ])


@dataclass
class FeaturesConfig:
    """Feature engineering configuration"""
    exclude_columns: list = field(default_factory=lambda: [
        "client_id", "target", "treatment", "rfm_segment",
        "rfm_score", "segment", "r_score", "f_score", "m_score"
    ])
    scaling: bool = True
    handle_missing: str = "median"


class Config:
    """
    Central configuration manager
    
    Loads configuration from YAML file and provides easy access
    to all settings.
    
    Examples
    --------
    >>> from src.config import Config
    >>> 
    >>> # Load default config
    >>> config = Config.load()
    >>> 
    >>> # Access settings
    >>> print(config.campaign.margin)
    >>> print(config.paths.rfm_data)
    >>> 
    >>> # Load with environment override
    >>> config = Config.load(environment='production')
    >>> 
    >>> # Access nested settings
    >>> print(config.model.xgboost['max_depth'])
    """
    
    def __init__(
        self,
        paths: Optional[PathConfig] = None,
        campaign: Optional[CampaignConfig] = None,
        model: Optional[ModelConfig] = None,
        metrics: Optional[MetricsConfig] = None,
        calibration: Optional[CalibrationConfig] = None,
        plotting: Optional[PlottingConfig] = None,
        features: Optional[FeaturesConfig] = None,
        raw_config: Optional[Dict] = None
    ):
        self.paths = paths or PathConfig()
        self.campaign = campaign or CampaignConfig()
        self.model = model or ModelConfig()
        self.metrics = metrics or MetricsConfig()
        self.calibration = calibration or CalibrationConfig()
        self.plotting = plotting or PlottingConfig()
        self.features = features or FeaturesConfig()
        self._raw = raw_config or {}
    
    @classmethod
    def load(
        cls,
        config_path: str = "config.yaml",
        environment: Optional[str] = None
    ) -> 'Config':
        """
        Load configuration from YAML file
        
        Parameters
        ----------
        config_path : str, optional
            Path to config YAML file. Default: 'config.yaml'
        environment : str, optional
            Environment name for overrides ('development', 'production')
        
        Returns
        -------
        config : Config
            Loaded configuration object
        """
        # Load YAML
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return cls()
        
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Apply environment overrides
        if environment and 'environments' in raw_config:
            if environment in raw_config['environments']:
                env_overrides = raw_config['environments'][environment]
                raw_config = cls._deep_merge(raw_config, env_overrides)
        
        # Parse sections
        paths_dict = raw_config.get('paths', {})
        campaign_dict = raw_config.get('campaign', {})
        model_dict = raw_config.get('model', {})
        metrics_dict = raw_config.get('metrics', {})
        calibration_dict = raw_config.get('calibration', {})
        plotting_dict = raw_config.get('plotting', {})
        features_dict = raw_config.get('features', {})
        
        # Create config objects
        paths = PathConfig(**paths_dict)
        campaign = CampaignConfig(**campaign_dict)
        model = ModelConfig(
            random_state=model_dict.get('random_state', 42),
            test_size=model_dict.get('test_size', 0.20),
            xgboost=model_dict.get('xgboost', {})
        )
        metrics = MetricsConfig(**metrics_dict)
        calibration = CalibrationConfig(**calibration_dict)
        
        # Handle plotting figsize tuple
        if 'figsize' in plotting_dict:
            plotting_dict['figsize'] = tuple(plotting_dict['figsize'])
        plotting = PlottingConfig(**plotting_dict)
        
        features = FeaturesConfig(**features_dict)
        
        return cls(
            paths=paths,
            campaign=campaign,
            model=model,
            metrics=metrics,
            calibration=calibration,
            plotting=plotting,
            features=features,
            raw_config=raw_config
        )
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def ensure_dirs(self):
        """Ensure all configured directories exist"""
        self.paths.ensure_dirs()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get raw configuration value by dot-notation key
        
        Examples
        --------
        >>> config.get('campaign.margin')
        50.0
        >>> config.get('model.xgboost.max_depth')
        5
        """
        keys = key.split('.')
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  campaign: margin=${self.campaign.margin}, "
            f"budget=${self.campaign.budget}\n"
            f"  model: random_state={self.model.random_state}\n"
            f"  paths: {len(vars(self.paths))} configured\n"
            f")"
        )


# Global config instance (lazy loaded)
_global_config: Optional[Config] = None


def get_config(reload: bool = False, environment: Optional[str] = None) -> Config:
    """
    Get global configuration instance
    
    Parameters
    ----------
    reload : bool, optional
        Force reload from file. Default: False
    environment : str, optional
        Environment name for overrides
    
    Returns
    -------
    config : Config
        Global configuration object
    
    Examples
    --------
    >>> from src.config import get_config
    >>> 
    >>> config = get_config()
    >>> print(config.campaign.margin)
    """
    global _global_config
    
    if _global_config is None or reload:
        # Detect environment from env variable if not specified
        if environment is None:
            environment = os.environ.get('UPLIFT_ENV', 'development')
        
        _global_config = Config.load(environment=environment)
        _global_config.ensure_dirs()
    
    return _global_config


# Convenience function
def load_config(config_path: str = "config.yaml", environment: Optional[str] = None) -> Config:
    """
    Load configuration from file
    
    Alias for Config.load()
    """
    return Config.load(config_path, environment)