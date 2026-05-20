"""
Physi-Cast Source Module
Physics-Informed Neural Networks for Climate Forecasting
"""

__version__ = "1.0.0"
__author__ = "Physi-Cast Development Team"

from .physics import PhysicsConstraints
from .network import FCNN, ResidualFCNN, PINNModel
from .trainer import PINNTrainer
from .utils import GeoNormalizer, DataDownscaler, SyntheticDataGenerator, create_training_dataset

__all__ = [
    'PhysicsConstraints',
    'FCNN',
    'ResidualFCNN',
    'PINNModel',
    'PINNTrainer',
    'GeoNormalizer',
    'DataDownscaler',
    'SyntheticDataGenerator',
    'create_training_dataset'
]
