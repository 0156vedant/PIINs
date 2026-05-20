"""
Neural Network Architecture for Physics-Informed Neural Networks (PINNs)
Implements a fully connected deep network for climate prediction
"""

import torch
import torch.nn as nn
import numpy as np


class FCNN(nn.Module):
    """
    Fully Connected Neural Network for PINN
    Maps (x, y, z, t) → (u, v, w, p, T)
    5 outputs: 3 velocity components, pressure, temperature
    """
    
    def __init__(self, input_dim=4, output_dim=5, hidden_layers=None, 
                 hidden_units=128, activation='tanh', use_batch_norm=False):
        """
        Initialize FCNN architecture
        
        Args:
            input_dim: Dimension of input (x, y, z, t)
            output_dim: Dimension of output (u, v, w, p, T)
            hidden_layers: List of hidden layer sizes
            hidden_units: Default hidden unit size if hidden_layers not specified
            activation: Activation function ('tanh', 'relu', 'gelu')
            use_batch_norm: Whether to use batch normalization
        """
        super(FCNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
        # Default architecture if not specified
        if hidden_layers is None:
            hidden_layers = [hidden_units] * 6  # 6 hidden layers
        
        self.hidden_layers = hidden_layers
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(self.activation)
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def predict(self, x, return_numpy=True):
        """
        Inference mode prediction
        
        Args:
            x: Input data (can be numpy or tensor)
            return_numpy: Return as numpy array
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            
            predictions = self.forward(x)
            
            if return_numpy:
                predictions = predictions.numpy()
        
        return predictions


class ResidualFCNN(nn.Module):
    """
    Enhanced FCNN with Residual connections
    Improves gradient flow and training stability
    """
    
    def __init__(self, input_dim=4, output_dim=5, hidden_layers=None,
                 hidden_units=128, activation='tanh'):
        """
        Initialize Residual FCNN
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_layers: List of hidden layer sizes
            hidden_units: Default hidden units
            activation: Activation function
        """
        super(ResidualFCNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_layers is None:
            hidden_layers = [hidden_units] * 6
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            block = ResidualBlock(hidden_layers[i], hidden_layers[i + 1], 
                                 activation=activation)
            self.residual_blocks.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """Forward pass with residual connections"""
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    """Single residual block"""
    
    def __init__(self, in_dim, out_dim, activation='tanh'):
        super(ResidualBlock, self).__init__()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
        # Skip connection projection if dimensions don't match
        self.skip_projection = None
        if in_dim != out_dim:
            self.skip_projection = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        """Forward with residual connection"""
        residual = x
        
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        
        x = x + residual
        x = self.activation(x)
        
        return x


class PINNModel:
    """
    Wrapper class for PINN training and inference
    Combines network, physics constraints, and data
    """
    
    def __init__(self, network=None, physics_constraints=None, device='cpu'):
        """
        Initialize PINN model
        
        Args:
            network: Neural network module
            physics_constraints: Physics constraint object
            device: Device for computation ('cpu' or 'cuda')
        """
        if network is None:
            network = FCNN(input_dim=4, output_dim=5, hidden_units=128)
        
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.physics = physics_constraints
        
    def forward(self, x):
        """Forward pass"""
        return self.network(x)
    
    def compute_loss(self, x_train, y_train, x_collocation, 
                     x_bc=None, y_bc=None, 
                     lambda_data=0.5, lambda_physics=0.5):
        """
        Compute combined data + physics loss
        
        Args:
            x_train: Training input coordinates
            y_train: Training labels
            x_collocation: Collocation points for physics loss
            x_bc: Boundary condition coordinates
            y_bc: Boundary condition values
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss
            
        Returns:
            Total loss, data loss, physics loss
        """
        # Data loss
        y_pred_train = self.network(x_train)
        data_loss = torch.mean((y_pred_train - y_train) ** 2)
        
        # Physics loss at collocation points
        x_collocation.requires_grad_(True)
        y_pred_collocation = self.network(x_collocation)
        physics_losses = self.physics.total_physics_loss(
            x_collocation, x_collocation, y_pred_collocation, y_bc
        )
        physics_loss = physics_losses['total']
        
        # Combined loss
        total_loss = lambda_data * data_loss + lambda_physics * physics_loss
        
        return total_loss, data_loss, physics_loss, physics_losses
    
    def to_device(self, device):
        """Move model to device"""
        self.device = torch.device(device)
        self.network = self.network.to(self.device)
