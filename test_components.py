"""
Demo and Testing Script for Physi-Cast
Tests individual components and generates demo predictions
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from physics import PhysicsConstraints
from network import FCNN, PINNModel
from utils import GeoNormalizer, SyntheticDataGenerator, create_training_dataset
from config import get_config


def test_physics_constraints():
    """Test physics constraint computation"""
    print("\n" + "="*80)
    print("TEST 1: Physics Constraints")
    print("="*80)
    
    physics = PhysicsConstraints()
    print("✓ PhysicsConstraints initialized")
    
    # Test with dummy data
    x = torch.randn(10, 4, requires_grad=True)
    y = torch.randn(10, 5, requires_grad=True)
    
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {y.shape}")
    
    # Test loss computation
    try:
        losses = physics.total_physics_loss(x, x, y)
        print(f"✓ Physics loss computation successful")
        print(f"  - Total loss: {losses['total'].item():.6e}")
        print(f"  - NS loss: {losses['ns'].item():.6e}")
        print(f"  - Thermal loss: {losses['thermal'].item():.6e}")
        print(f"  - Continuity loss: {losses['continuity'].item():.6e}")
    except Exception as e:
        print(f"✗ Physics loss computation failed: {str(e)}")
        return False
    
    return True


def test_neural_network():
    """Test neural network architecture"""
    print("\n" + "="*80)
    print("TEST 2: Neural Network Architecture")
    print("="*80)
    
    model = FCNN(input_dim=4, output_dim=5, hidden_units=128)
    print("✓ FCNN model initialized")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(32, 4)
    try:
        y = model(x)
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {y.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        return False
    
    # Test prediction mode
    try:
        x_np = np.random.randn(10, 4)
        y_np = model.predict(x_np, return_numpy=True)
        print(f"✓ Prediction mode successful")
        print(f"  - Prediction shape: {y_np.shape}")
    except Exception as e:
        print(f"✗ Prediction mode failed: {str(e)}")
        return False
    
    return True


def test_data_processing():
    """Test data processing utilities"""
    print("\n" + "="*80)
    print("TEST 3: Data Processing")
    print("="*80)
    
    # Test normalizer
    normalizer = GeoNormalizer()
    print("✓ GeoNormalizer initialized")
    
    x = np.array([25.0, 50.0])
    x_norm = normalizer.normalize(x, 'x')
    x_denorm = normalizer.denormalize(x_norm, 'x')
    
    print(f"  - Original: {x}")
    print(f"  - Normalized: {x_norm}")
    print(f"  - Denormalized: {x_denorm}")
    print(f"✓ Normalization working correctly")
    
    # Test data generation
    generator = SyntheticDataGenerator(domain_size=50, time_steps=24)
    print("\n✓ SyntheticDataGenerator initialized")
    
    try:
        T = generator.generate_temperature_field()
        print(f"  - Temperature field shape: {T.shape}")
        
        u, v, w = generator.generate_velocity_field()
        print(f"  - Velocity field shapes: u={u.shape}, v={v.shape}, w={w.shape}")
        
        P = generator.generate_pressure_field()
        print(f"  - Pressure field shape: {P.shape}")
        
        colloc_points = generator.get_collocation_points(1000)
        print(f"  - Collocation points: {colloc_points.shape}")
        
        print("✓ Data generation successful")
    except Exception as e:
        print(f"✗ Data generation failed: {str(e)}")
        return False
    
    # Test training dataset creation
    try:
        X_train, y_train, norm = create_training_dataset(1000)
        print(f"\n✓ Training dataset created")
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - y_train shape: {y_train.shape}")
    except Exception as e:
        print(f"✗ Training dataset creation failed: {str(e)}")
        return False
    
    return True


def test_pinn_model():
    """Test PINN model wrapper"""
    print("\n" + "="*80)
    print("TEST 4: PINN Model Wrapper")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    network = FCNN(input_dim=4, output_dim=5)
    physics = PhysicsConstraints()
    model = PINNModel(network=network, physics_constraints=physics, device=device)
    
    print(f"✓ PINNModel initialized on device: {device}")
    
    # Test loss computation
    x_train = torch.randn(32, 4).to(device)
    y_train = torch.randn(32, 5).to(device)
    x_colloc = torch.randn(100, 4).to(device)
    
    try:
        total_loss, data_loss, physics_loss, losses_dict = model.compute_loss(
            x_train, y_train, x_colloc,
            lambda_data=0.5, lambda_physics=0.5
        )
        
        print(f"✓ Loss computation successful")
        print(f"  - Total loss: {total_loss.item():.6e}")
        print(f"  - Data loss: {data_loss.item():.6e}")
        print(f"  - Physics loss: {physics_loss.item():.6e}")
    except Exception as e:
        print(f"✗ Loss computation failed: {str(e)}")
        return False
    
    return True


def demo_predictions():
    """Generate demo predictions"""
    print("\n" + "="*80)
    print("DEMO: Sample Predictions")
    print("="*80)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FCNN(input_dim=4, output_dim=5).to(device)
    model.eval()
    
    print(f"Using device: {device}\n")
    
    # Create sample coordinate points
    locations = [
        {"name": "New York (Sea Level, Noon)", "x": 40.7128, "y": -74.0060, "z": 0, "t": 43200},
        {"name": "Mountain Peak (5000m, Dawn)", "x": 40.0, "y": -105.0, "z": 5000, "t": 21600},
        {"name": "Valley (500m, Dusk)", "x": 39.0, "y": -106.0, "z": 500, "t": 64800},
        {"name": "Desert (100m, Midday)", "x": 35.0, "y": -115.0, "z": 100, "t": 43200},
    ]
    
    print("Sample Predictions:")
    print("-" * 80)
    
    normalizer = GeoNormalizer()
    
    for loc in locations:
        # Normalize coordinates
        x_norm, y_norm, z_norm, t_norm = normalizer.normalize_coords(
            np.array([loc["x"]]),
            np.array([loc["y"]]),
            np.array([loc["z"]]),
            np.array([loc["t"]])
        )
        
        # Create input tensor
        input_tensor = torch.tensor(
            [[x_norm[0], y_norm[0], z_norm[0], t_norm[0]]],
            dtype=torch.float32,
            device=device
        )
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor).cpu().numpy()[0]
        
        u, v, w, p, T = output
        
        # Denormalize pressure and temperature (rough)
        p_phys = p * 50000 + 101325
        T_phys = T * 10 + 288.15
        
        # Calculate wind speed
        wind_speed = np.sqrt(u**2 + v**2 + w**2)
        
        print(f"\n{loc['name']}")
        print(f"  Location: ({loc['x']:.2f}°, {loc['y']:.2f}°)")
        print(f"  Altitude: {loc['z']} m, Time: {loc['t']/3600:.1f}h")
        print(f"  Wind (U, V, W): ({u:7.3f}, {v:7.3f}, {w:7.3f}) m/s")
        print(f"  Wind Speed: {wind_speed:.2f} m/s ({wind_speed*3.6:.1f} km/h)")
        print(f"  Pressure: {p_phys/100:.1f} hPa")
        print(f"  Temperature: {T_phys - 273.15:.1f}°C ({T_phys:.2f} K)")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("PHYSI-CAST COMPONENT TESTS")
    print("="*80)
    
    tests = [
        ("Physics Constraints", test_physics_constraints),
        ("Neural Network", test_neural_network),
        ("Data Processing", test_data_processing),
        ("PINN Model Wrapper", test_pinn_model),
        ("Demo Predictions", demo_predictions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} - Exception: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    return all(p for _, p in results)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physi-Cast Component Tests"
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--physics', action='store_true', help='Test physics')
    parser.add_argument('--network', action='store_true', help='Test network')
    parser.add_argument('--data', action='store_true', help='Test data')
    parser.add_argument('--pinn', action='store_true', help='Test PINN model')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')
    
    args = parser.parse_args()
    
    if args.all or not any([args.physics, args.network, args.data, args.pinn, args.demo]):
        run_all_tests()
    else:
        if args.physics:
            test_physics_constraints()
        if args.network:
            test_neural_network()
        if args.data:
            test_data_processing()
        if args.pinn:
            test_pinn_model()
        if args.demo:
            demo_predictions()


if __name__ == '__main__':
    main()
