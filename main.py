"""
Main Training Script for Physi-Cast PINN
Orchestrates model training pipeline
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from physics import PhysicsConstraints
from network import FCNN, PINNModel
from trainer import PINNTrainer
from utils import GeoNormalizer, create_training_dataset, SyntheticDataGenerator


def main(args):
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("PHYSI-CAST: PINN CLIMATE FORECASTING SYSTEM")
    print("Physics-Informed Neural Network Training")
    print("="*80 + "\n")
    
    # ===========================================================================
    # SETUP
    # ===========================================================================
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # Paths
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # ===========================================================================
    # DATA PREPARATION
    # ===========================================================================
    
    print("\n[STAGE 0] Data Preparation")
    print("-" * 80)
    
    print("[DATA] Initializing normalizer...")
    normalizer = GeoNormalizer()
    
    print("[DATA] Generating synthetic training dataset...")
    print(f"[DATA] Generating {args.n_train_samples} training samples...")
    
    X_train, y_train, normalizer = create_training_dataset(
        n_samples=args.n_train_samples,
        normalizer=normalizer
    )
    
    print(f"[DATA] Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Generate collocation points for physics loss
    print(f"[DATA] Generating {args.n_collocation} collocation points...")
    generator = SyntheticDataGenerator()
    x_collocation = generator.get_collocation_points(args.n_collocation)
    
    # Normalize collocation points
    x_col_norm, y_col_norm, z_col_norm, t_col_norm = normalizer.normalize_coords(
        x_collocation[:, 0],
        x_collocation[:, 1],
        x_collocation[:, 2],
        x_collocation[:, 3]
    )
    x_collocation = np.column_stack([x_col_norm, y_col_norm, z_col_norm, t_col_norm])
    
    print(f"[DATA] Collocation data shape: {x_collocation.shape}")
    print("[DATA] ✓ Data preparation complete\n")
    
    # ===========================================================================
    # MODEL INITIALIZATION
    # ===========================================================================
    
    print("[MODEL] Initializing PINN...")
    
    # Neural network
    network = FCNN(
        input_dim=4,
        output_dim=5,
        hidden_layers=[128, 128, 128, 128, 128, 128],
        activation='tanh'
    )
    
    print(f"[MODEL] Network Architecture:")
    print(f"        Input: 4 (x, y, z, t)")
    print(f"        Hidden: 6 layers × 128 units")
    print(f"        Output: 5 (u, v, w, p, T)")
    print(f"[MODEL] Total parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Physics constraints
    physics = PhysicsConstraints()
    print("[MODEL] Physics constraints: Navier-Stokes, Thermal Diffusion, Continuity")
    
    # PINN model wrapper
    model = PINNModel(network=network, physics_constraints=physics, device=device)
    
    print("[MODEL] ✓ Model initialization complete\n")
    
    # ===========================================================================
    # TRAINING
    # ===========================================================================
    
    print("[TRAINING] Initializing trainer...")
    trainer = PINNTrainer(model, physics, device=device)
    
    # Full training pipeline
    trainer.fit(
        X_train=X_train,
        y_train=y_train,
        x_collocation=x_collocation,
        x_bc=None,
        y_bc=None,
        batch_size=args.batch_size,
        adam_epochs=args.adam_epochs,
        adam_lr=args.learning_rate,
        lbfgs_iter=args.lbfgs_iter,
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        verbose=True,
        save_path=str(models_dir / 'pinn_model.pth')
    )
    
    print("[TRAINING] ✓ Training complete\n")
    
    # ===========================================================================
    # EVALUATION & VISUALIZATION
    # ===========================================================================
    
    print("[EVALUATION] Evaluating model performance...")
    
    # Test on unseen points
    n_test = 1000
    x_test, y_test, _ = create_training_dataset(n_test, normalizer)
    
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
    model.network.eval()
    with torch.no_grad():
        y_pred = model.network(x_test_tensor)
    
    mse_loss = torch.mean((y_pred - y_test_tensor) ** 2).item()
    rmse_loss = np.sqrt(mse_loss)
    
    print(f"[EVALUATION] Test MSE: {mse_loss:.6e}")
    print(f"[EVALUATION] Test RMSE: {rmse_loss:.6e}")
    
    # Save training plots
    print("[EVALUATION] Saving training history plots...")
    trainer.plot_history(save_path=str(results_dir / 'training_history.png'))
    print("[EVALUATION] ✓ Plots saved\n")
    
    # ===========================================================================
    # SAVE ARTIFACTS
    # ===========================================================================
    
    print("[ARTIFACTS] Saving model artifacts...")
    
    # Already saved model weights, now save normalizer
    import pickle
    with open(models_dir / 'normalizer.pkl', 'wb') as f:
        pickle.dump(normalizer, f)
    
    # Save training configuration
    config = {
        'architecture': {
            'input_dim': 4,
            'output_dim': 5,
            'hidden_layers': [128, 128, 128, 128, 128, 128],
            'activation': 'tanh'
        },
        'training': {
            'n_train_samples': args.n_train_samples,
            'n_collocation': args.n_collocation,
            'batch_size': args.batch_size,
            'adam_epochs': args.adam_epochs,
            'adam_lr': args.learning_rate,
            'lbfgs_iter': args.lbfgs_iter,
            'lambda_data': args.lambda_data,
            'lambda_physics': args.lambda_physics
        },
        'performance': {
            'test_mse': float(mse_loss),
            'test_rmse': float(rmse_loss)
        }
    }
    
    import json
    with open(results_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[ARTIFACTS] Model saved to: models/pinn_model.pth")
    print(f"[ARTIFACTS] Normalizer saved to: models/normalizer.pkl")
    print(f"[ARTIFACTS] Config saved to: results/training_config.json")
    print("[ARTIFACTS] ✓ Artifacts saved\n")
    
    # ===========================================================================
    # COMPLETION
    # ===========================================================================
    
    print("="*80)
    print("TRAINING PIPELINE COMPLETE ✓")
    print("="*80)
    
    print("\n[NEXT STEPS]")
    print("1. Start the FastAPI backend:")
    print("   python -m uvicorn api.server:app --reload --port 8000")
    print("\n2. Start the Streamlit dashboard (in another terminal):")
    print("   streamlit run dashboard/app.py")
    print("\n3. Access the system:")
    print("   - API: http://localhost:8000")
    print("   - Dashboard: http://localhost:8501")
    print("   - API Docs: http://localhost:8000/docs")
    print("\n" + "="*80 + "\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Physi-Cast PINN Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Train with default parameters
  python main.py --n-train-samples 20000           # Use more training samples
  python main.py --adam-epochs 200 --lbfgs-iter 50 # Adjust optimization stages
  python main.py --lambda-physics 0.6               # Increase physics weight
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--n-train-samples',
        type=int,
        default=5000,
        help='Number of training samples (default: 5000)'
    )
    
    parser.add_argument(
        '--n-collocation',
        type=int,
        default=5000,
        help='Number of collocation points for physics loss (default: 5000)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for Adam optimizer (default: 32)'
    )
    
    parser.add_argument(
        '--adam-epochs',
        type=int,
        default=100,
        help='Number of Adam training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Adam learning rate (default: 1e-3)'
    )
    
    parser.add_argument(
        '--lbfgs-iter',
        type=int,
        default=50,
        help='Number of L-BFGS iterations (default: 50)'
    )
    
    # Loss weights
    parser.add_argument(
        '--lambda-data',
        type=float,
        default=0.5,
        help='Weight for data loss (default: 0.5)'
    )
    
    parser.add_argument(
        '--lambda-physics',
        type=float,
        default=0.5,
        help='Weight for physics loss (default: 0.5)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
