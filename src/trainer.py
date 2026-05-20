"""
Training Module for Physics-Informed Neural Networks
Implements Adam + L-BFGS two-stage optimization
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


class PINNTrainer:
    """
    Two-stage trainer for PINN models
    Stage 1: Adam optimizer (fast initial convergence)
    Stage 2: L-BFGS optimizer (fine-tuning physics residuals)
    """
    
    def __init__(self, model, physics, device='cpu'):
        """
        Initialize trainer
        
        Args:
            model: PINN model instance
            physics: Physics constraints
            device: Device for computation
        """
        self.model = model
        self.physics = physics
        self.device = device
        
        self.training_history = {
            'epoch': [],
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'ns_loss': [],
            'thermal_loss': [],
            'continuity_loss': []
        }
        
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def prepare_data_loader(self, X_train, y_train, batch_size=32, shuffle=True):
        """
        Create data loader
        
        Args:
            X_train: Training coordinates
            y_train: Training labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        X_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_tensor = torch.from_numpy(y_train).float().to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def train_stage1_adam(self, train_loader, x_collocation, x_bc=None, y_bc=None,
                         epochs=100, learning_rate=1e-3, lambda_data=0.5, 
                         lambda_physics=0.5, verbose=True):
        """
        Stage 1: Adam optimizer for initial convergence
        
        Args:
            train_loader: Training data loader
            x_collocation: Collocation points for physics loss
            x_bc: Boundary condition points
            y_bc: Boundary condition values
            epochs: Number of epochs
            learning_rate: Learning rate
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss
            verbose: Print training progress
        """
        optimizer = optim.Adam(self.model.network.parameters(), 
                              lr=learning_rate)
        
        x_collocation = torch.from_numpy(x_collocation).float().to(self.device)
        
        if x_bc is not None:
            x_bc = torch.from_numpy(x_bc).float().to(self.device)
        if y_bc is not None:
            y_bc = torch.from_numpy(y_bc).float().to(self.device)
        
        self.model.network.train()
        
        if verbose:
            pbar = tqdm(range(epochs), desc="Stage 1 (Adam)")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            n_batches = 0
            
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Compute loss
                x_batch.requires_grad_(True)
                y_batch.requires_grad_(False)
                
                y_pred_batch = self.model.forward(x_batch)
                data_loss = torch.mean((y_pred_batch - y_batch) ** 2)
                
                # Physics loss on collocation points
                x_collocation.requires_grad_(True)
                y_pred_collocation = self.model.forward(x_collocation)
                physics_losses = self.physics.total_physics_loss(
                    x_collocation, x_collocation, y_pred_collocation, y_bc
                )
                physics_loss = physics_losses['total']
                
                # Total loss
                total_loss = lambda_data * data_loss + lambda_physics * physics_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), 1.0)
                optimizer.step()
                
                # Record metrics
                epoch_loss += total_loss.item()
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()
                n_batches += 1
            
            # Average over batches
            epoch_loss /= n_batches
            epoch_data_loss /= n_batches
            epoch_physics_loss /= n_batches
            
            # Record history
            self.training_history['epoch'].append(epoch)
            self.training_history['total_loss'].append(epoch_loss)
            self.training_history['data_loss'].append(epoch_data_loss)
            self.training_history['physics_loss'].append(epoch_physics_loss)
            
            # Save best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_model_state = self.model.network.state_dict().copy()
            
            if verbose:
                pbar.set_postfix({
                    'total_loss': f'{epoch_loss:.6f}',
                    'data_loss': f'{epoch_data_loss:.6f}',
                    'physics_loss': f'{epoch_physics_loss:.6f}'
                })
    
    def train_stage2_lbfgs(self, train_loader, x_collocation, x_bc=None, y_bc=None,
                          max_iter=100, lambda_data=0.5, lambda_physics=0.5,
                          verbose=True):
        """
        Stage 2: L-BFGS optimizer for fine-tuning physics residuals
        
        Args:
            train_loader: Training data loader
            x_collocation: Collocation points
            x_bc: Boundary condition points
            y_bc: Boundary condition values
            max_iter: Maximum iterations
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss
            verbose: Print training progress
        """
        # Collect all data for LBFGS (doesn't support mini-batch well)
        X_full = []
        y_full = []
        for x_batch, y_batch in train_loader:
            X_full.append(x_batch)
            y_full.append(y_batch)
        
        X_full = torch.cat(X_full, dim=0).requires_grad_(True)
        y_full = torch.cat(y_full, dim=0)
        
        x_collocation = torch.from_numpy(x_collocation).float().to(self.device)
        x_collocation.requires_grad_(True)
        
        if x_bc is not None:
            x_bc = torch.from_numpy(x_bc).float().to(self.device)
        if y_bc is not None:
            y_bc = torch.from_numpy(y_bc).float().to(self.device)
        
        optimizer = optim.LBFGS(self.model.network.parameters(), 
                               max_iter=max_iter, line_search_fn='strong_wolfe')
        
        self.model.network.train()
        
        iteration = 0
        
        def closure():
            nonlocal iteration
            
            optimizer.zero_grad()
            
            # Data loss
            y_pred_full = self.model.forward(X_full)
            data_loss = torch.mean((y_pred_full - y_full) ** 2)
            
            # Physics loss
            y_pred_collocation = self.model.forward(x_collocation)
            physics_losses = self.physics.total_physics_loss(
                x_collocation, x_collocation, y_pred_collocation, y_bc
            )
            physics_loss = physics_losses['total']
            
            # Total loss
            total_loss = lambda_data * data_loss + lambda_physics * physics_loss
            
            total_loss.backward()
            
            if verbose and iteration % 10 == 0:
                print(f"L-BFGS Iter {iteration:3d}: Loss = {total_loss.item():.6e}")
            
            iteration += 1
            
            return total_loss
        
        optimizer.step(closure)
    
    def fit(self, X_train, y_train, x_collocation, x_bc=None, y_bc=None,
           batch_size=32, adam_epochs=100, adam_lr=1e-3, 
           lbfgs_iter=100, lambda_data=0.5, lambda_physics=0.5,
           verbose=True, save_path=None):
        """
        Full training pipeline: Adam -> L-BFGS
        
        Args:
            X_train: Training coordinates
            y_train: Training labels
            x_collocation: Collocation points
            x_bc: Boundary condition points
            y_bc: Boundary condition values
            batch_size: Batch size for Adam
            adam_epochs: Number of Adam epochs
            adam_lr: Adam learning rate
            lbfgs_iter: L-BFGS iterations
            lambda_data: Data loss weight
            lambda_physics: Physics loss weight
            verbose: Print progress
            save_path: Path to save model
        """
        print("\n" + "="*60)
        print("PHYSI-CAST PINN TRAINING")
        print("="*60)
        
        # Stage 1: Adam
        print("\n[STAGE 1] Adam Optimizer (Initial Convergence)")
        print("-" * 60)
        
        train_loader = self.prepare_data_loader(X_train, y_train, batch_size, 
                                               shuffle=True)
        
        self.train_stage1_adam(train_loader, x_collocation, x_bc, y_bc,
                              adam_epochs, adam_lr, lambda_data, 
                              lambda_physics, verbose)
        
        # Stage 2: L-BFGS
        print("\n[STAGE 2] L-BFGS Optimizer (Physics Refinement)")
        print("-" * 60)
        
        train_loader = self.prepare_data_loader(X_train, y_train, batch_size,
                                               shuffle=False)
        
        self.train_stage2_lbfgs(train_loader, x_collocation, x_bc, y_bc,
                               lbfgs_iter, lambda_data, lambda_physics, verbose)
        
        # Save model
        if save_path:
            self.save_model(save_path)
            print(f"\nModel saved to {save_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.network.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        self.model.network.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_training_history(self):
        """Get training history as dictionary"""
        return self.training_history
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(self.training_history['epoch'], 
                        self.training_history['total_loss'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Data loss
        axes[0, 1].plot(self.training_history['epoch'],
                        self.training_history['data_loss'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].set_title('Data Loss')
        axes[0, 1].grid(True)
        
        # Physics loss
        axes[1, 0].plot(self.training_history['epoch'],
                        self.training_history['physics_loss'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].set_title('Physics Loss')
        axes[1, 0].grid(True)
        
        # Log scale
        axes[1, 1].semilogy(self.training_history['epoch'],
                            self.training_history['total_loss'], label='Total')
        axes[1, 1].semilogy(self.training_history['epoch'],
                            self.training_history['data_loss'], label='Data')
        axes[1, 1].semilogy(self.training_history['epoch'],
                            self.training_history['physics_loss'], label='Physics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].set_title('Loss Comparison (Log Scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return fig
