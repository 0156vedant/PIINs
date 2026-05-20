"""
Physi-Cast Configuration File
Centralized configuration for the entire system
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


# ==============================================================================
# PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    """Neural Network Configuration"""
    input_dim: int = 4  # x, y, z, t
    output_dim: int = 5  # u, v, w, p, T
    hidden_layers: tuple = (128, 128, 128, 128, 128, 128)
    activation: str = 'tanh'
    use_batch_norm: bool = False
    use_residual: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert len(self.hidden_layers) > 0, "hidden_layers must not be empty"
        assert self.activation in ['tanh', 'relu', 'gelu'], "Invalid activation"


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """Training Hyperparameters"""
    n_train_samples: int = 5000
    n_collocation: int = 5000
    batch_size: int = 32
    adam_epochs: int = 100
    adam_lr: float = 1e-3
    adam_weight_decay: float = 1e-5
    lbfgs_iter: int = 50
    lbfgs_line_search: str = 'strong_wolfe'
    lambda_data: float = 0.5
    lambda_physics: float = 0.5
    grad_clip: float = 1.0
    device: str = 'cuda'  # 'cuda' or 'cpu'
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_train_samples > 0, "n_train_samples must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.lambda_data <= 1, "lambda_data must be in [0, 1]"
        assert 0 <= self.lambda_physics <= 1, "lambda_physics must be in [0, 1]"


# ==============================================================================
# PHYSICS CONFIGURATION
# ==============================================================================

@dataclass
class PhysicsConfig:
    """Physical Constants"""
    # Air properties (sea level, standard atmosphere)
    nu: float = 1.5e-5  # Kinematic viscosity (m²/s)
    alpha: float = 2.2e-5  # Thermal diffusivity (m²/s)
    rho: float = 1.225  # Air density (kg/m³)
    g: float = 9.81  # Gravitational acceleration (m/s²)
    
    # Loss weights for physics constraints
    w_ns: float = 0.4  # Navier-Stokes weight
    w_thermal: float = 0.3  # Thermal diffusion weight
    w_continuity: float = 0.2  # Continuity weight
    w_bc: float = 0.1  # Boundary condition weight
    
    # Domain bounds
    x_bounds: tuple = (0, 100)  # Longitude in km
    y_bounds: tuple = (0, 100)  # Latitude in km
    z_bounds: tuple = (0, 5000)  # Altitude in m
    t_bounds: tuple = (0, 86400)  # Time in seconds
    
    def __post_init__(self):
        """Validate configuration"""
        w_sum = self.w_ns + self.w_thermal + self.w_continuity + self.w_bc
        assert abs(w_sum - 1.0) < 1e-6, "Physics loss weights must sum to 1"


# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

@dataclass
class DataConfig:
    """Data Processing Configuration"""
    # Downscaling
    coarse_resolution: int = 28000  # ERA5 resolution in meters
    fine_resolution: int = 100  # Target fine resolution in meters
    
    # Normalization
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    
    # Data augmentation
    add_noise: bool = True
    noise_std: float = 0.01
    add_spatial_heterogeneity: bool = True
    heterogeneity_scale: float = 0.1


# ==============================================================================
# API CONFIGURATION
# ==============================================================================

@dataclass
class APIConfig:
    """FastAPI Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 4
    log_level: str = "info"
    
    # CORS
    allow_origins: list = None
    allow_credentials: bool = True
    allow_methods: list = None
    allow_headers: list = None
    
    def __post_init__(self):
        """Set defaults"""
        if self.allow_origins is None:
            self.allow_origins = ["*"]
        if self.allow_methods is None:
            self.allow_methods = ["*"]
        if self.allow_headers is None:
            self.allow_headers = ["*"]


# ==============================================================================
# DASHBOARD CONFIGURATION
# ==============================================================================

@dataclass
class DashboardConfig:
    """Streamlit Dashboard Configuration"""
    page_title: str = "Physi-Cast"
    page_icon: str = "🌤️"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # API connection
    api_url: str = "http://localhost:8000"
    api_timeout: int = 10
    
    # Map settings
    map_zoom: int = 12
    map_center_lat: float = 40.7128
    map_center_lon: float = -74.0060
    map_tiles: str = "OpenStreetMap"
    
    # Cache settings
    cache_ttl: int = 300  # seconds


# ==============================================================================
# COMPLETE CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Master Configuration"""
    model: ModelConfig = None
    training: TrainingConfig = None
    physics: PhysicsConfig = None
    data: DataConfig = None
    api: APIConfig = None
    dashboard: DashboardConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.dashboard is None:
            self.dashboard = DashboardConfig()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'physics': self.physics.__dict__,
            'data': self.data.__dict__,
            'api': self.api.__dict__,
            'dashboard': self.dashboard.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            data=DataConfig(**config_dict.get('data', {})),
            api=APIConfig(**config_dict.get('api', {})),
            dashboard=DashboardConfig(**config_dict.get('dashboard', {}))
        )
    
    def save(self, path: Path):
        """Save configuration to JSON"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from JSON"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# ==============================================================================
# DEFAULT CONFIGURATION INSTANCES
# ==============================================================================

# Default configuration
default_config = Config()

# For quick training
quick_train_config = Config(
    training=TrainingConfig(
        n_train_samples=2000,
        n_collocation=2000,
        adam_epochs=50,
        lbfgs_iter=20
    )
)

# For high-accuracy training
high_accuracy_config = Config(
    training=TrainingConfig(
        n_train_samples=10000,
        n_collocation=10000,
        adam_epochs=200,
        lbfgs_iter=100,
        lambda_physics=0.6
    )
)

# For CPU-only training
cpu_config = Config(
    training=TrainingConfig(device='cpu')
)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_config(preset: str = 'default') -> Config:
    """
    Get configuration preset
    
    Args:
        preset: 'default', 'quick', 'high_accuracy', or 'cpu'
    
    Returns:
        Configuration object
    """
    presets = {
        'default': default_config,
        'quick': quick_train_config,
        'high_accuracy': high_accuracy_config,
        'cpu': cpu_config
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Options: {list(presets.keys())}")
    
    return presets[preset]


if __name__ == '__main__':
    # Demo: Print default configuration
    config = get_config('default')
    
    print("=" * 80)
    print("PHYSI-CAST CONFIGURATION")
    print("=" * 80)
    print()
    
    print("Model Configuration:")
    for key, value in config.model.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in config.training.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nPhysics Configuration:")
    for key, value in config.physics.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nData Configuration:")
    for key, value in config.data.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nAPI Configuration:")
    for key, value in config.api.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nDashboard Configuration:")
    for key, value in config.dashboard.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
