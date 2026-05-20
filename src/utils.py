"""
Data Processing and Utilities for Physi-Cast
Handles geospatial coordinate normalization, downscaling, and data loading
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import torch


class GeoNormalizer:
    """
    Normalize geographical coordinates to [-1, 1] range
    Essential for PINN training stability
    """
    
    def __init__(self, bounds=None):
        """
        Initialize normalizer
        
        Args:
            bounds: Dict with 'x', 'y', 'z', 't' min/max values
                   e.g., {'x': (0, 100), 'y': (0, 100), 'z': (0, 5000), 't': (0, 86400)}
        """
        self.bounds = bounds or {
            'x': (0, 100),      # Horizontal distance in km
            'y': (0, 100),      # Horizontal distance in km
            'z': (0, 5000),     # Altitude in meters
            't': (0, 86400)     # Time in seconds (1 day)
        }
        self.means = {}
        self.scales = {}
        
        for key, (min_val, max_val) in self.bounds.items():
            self.means[key] = (min_val + max_val) / 2
            self.scales[key] = (max_val - min_val) / 2
    
    def normalize(self, data, key):
        """
        Normalize data to [-1, 1]
        
        Args:
            data: Data to normalize
            key: 'x', 'y', 'z', or 't'
            
        Returns:
            Normalized data
        """
        return (data - self.means[key]) / self.scales[key]
    
    def denormalize(self, data, key):
        """
        Denormalize data back to original scale
        
        Args:
            data: Normalized data
            key: 'x', 'y', 'z', or 't'
            
        Returns:
            Denormalized data
        """
        return data * self.scales[key] + self.means[key]
    
    def normalize_coords(self, x, y, z, t):
        """
        Normalize coordinate arrays
        
        Args:
            x, y, z, t: Coordinate arrays
            
        Returns:
            Normalized coordinates
        """
        x_norm = self.normalize(x, 'x')
        y_norm = self.normalize(y, 'y')
        z_norm = self.normalize(z, 'z')
        t_norm = self.normalize(t, 't')
        
        return x_norm, y_norm, z_norm, t_norm


class DataDownscaler:
    """
    Statistical downscaling from coarse grid to high resolution
    Converts ERA5 data (~28km) to farm-level resolution (~100m)
    """
    
    def __init__(self, coarse_resolution=28000, fine_resolution=100):
        """
        Initialize downscaler
        
        Args:
            coarse_resolution: Coarse grid resolution in meters
            fine_resolution: Target fine grid resolution in meters
        """
        self.coarse_res = coarse_resolution
        self.fine_res = fine_resolution
        self.scaling_factor = coarse_resolution / fine_resolution
    
    def downscale_bilinear(self, coarse_data, scale_factor):
        """
        Simple bilinear interpolation downscaling
        
        Args:
            coarse_data: Coarse resolution data (ny_coarse, nx_coarse)
            scale_factor: Scaling factor
            
        Returns:
            Downscaled data (ny_fine, nx_fine)
        """
        ny, nx = coarse_data.shape
        ny_fine = int(ny * scale_factor)
        nx_fine = int(nx * scale_factor)
        
        # Create coordinate grids
        y_coarse = np.linspace(0, 1, ny)
        x_coarse = np.linspace(0, 1, nx)
        y_fine = np.linspace(0, 1, ny_fine)
        x_fine = np.linspace(0, 1, nx_fine)
        
        # Use scipy griddata for interpolation
        points = np.array(np.meshgrid(x_coarse, y_coarse)).T.reshape(-1, 2)
        values = coarse_data.flatten()
        
        xi, yi = np.meshgrid(x_fine, y_fine)
        fine_coords = np.array([xi.flatten(), yi.flatten()]).T
        
        downscaled = griddata(points, values, fine_coords, method='linear')
        downscaled = downscaled.reshape(ny_fine, nx_fine)
        
        # Handle NaN values from extrapolation
        downscaled = np.nan_to_num(downscaled, nan=np.nanmean(values))
        
        return downscaled
    
    def add_topographic_correction(self, downscaled_temp, elevation_map, 
                                   lapse_rate=0.0065):
        """
        Correct temperature for local elevation changes
        Uses standard atmospheric lapse rate
        
        Args:
            downscaled_temp: Temperature field at sea level
            elevation_map: Local elevation map (meters)
            lapse_rate: Temperature lapse rate (K/m)
            
        Returns:
            Elevation-corrected temperature
        """
        reference_elevation = 0
        elevation_diff = elevation_map - reference_elevation
        temp_correction = lapse_rate * elevation_diff
        
        corrected_temp = downscaled_temp - temp_correction
        
        return corrected_temp
    
    def add_spatial_heterogeneity(self, downscaled_data, heterogeneity_scale=0.1):
        """
        Add realistic spatial variability to downscaled data
        Uses Gaussian smoothing + perturbation
        
        Args:
            downscaled_data: Base downscaled data
            heterogeneity_scale: Scale of perturbations
            
        Returns:
            Data with added spatial heterogeneity
        """
        from scipy.ndimage import gaussian_filter
        
        # Add small-scale noise
        noise = np.random.normal(0, heterogeneity_scale, downscaled_data.shape)
        
        # Smooth to create realistic patterns
        smoothed_noise = gaussian_filter(noise, sigma=2)
        
        result = downscaled_data + smoothed_noise
        
        return result


class SyntheticDataGenerator:
    """
    Generate synthetic atmospheric data for testing
    Based on simplified atmospheric profiles
    """
    
    def __init__(self, domain_size=50, time_steps=24, resolution=100):
        """
        Initialize synthetic data generator
        
        Args:
            domain_size: Spatial domain size in km
            time_steps: Number of time steps
            resolution: Grid resolution in meters
        """
        self.domain_size = domain_size * 1000  # Convert to meters
        self.time_steps = time_steps
        self.resolution = resolution
        
        self.nx = int(self.domain_size / resolution)
        self.ny = int(self.domain_size / resolution)
        self.nz = 50  # Vertical levels
    
    def generate_temperature_field(self, base_temp=288.15, diurnal_amplitude=5):
        """
        Generate synthetic temperature field with diurnal cycle
        
        Args:
            base_temp: Base temperature (K)
            diurnal_amplitude: Amplitude of diurnal variation (K)
            
        Returns:
            Temperature array (nx, ny, nz, time_steps)
        """
        T = np.zeros((self.nx, self.ny, self.nz, self.time_steps))
        
        for t in range(self.time_steps):
            # Diurnal cycle
            diurnal = diurnal_amplitude * np.sin(2 * np.pi * t / self.time_steps)
            
            # Vertical profile (lapse rate)
            for k in range(self.nz):
                height = k * 100  # 100m per level
                lapse = height * 0.0065  # 6.5 K/km
                
                # Spatial variation
                x = np.linspace(0, 1, self.nx)
                y = np.linspace(0, 1, self.ny)
                X, Y = np.meshgrid(x, y)
                
                spatial_var = 3 * np.sin(X * np.pi) * np.cos(Y * np.pi)
                
                T[:, :, k, t] = base_temp + diurnal + spatial_var - lapse
        
        return T
    
    def generate_velocity_field(self, wind_speed=5.0):
        """
        Generate synthetic wind field
        
        Args:
            wind_speed: Mean wind speed (m/s)
            
        Returns:
            Velocity arrays (u, v, w) shape: (nx, ny, nz, time_steps)
        """
        u = np.zeros((self.nx, self.ny, self.nz, self.time_steps))
        v = np.zeros((self.nx, self.ny, self.nz, self.time_steps))
        w = np.zeros((self.nx, self.ny, self.nz, self.time_steps))
        
        for t in range(self.time_steps):
            # Mean wind
            u[:, :, :, t] = wind_speed
            
            # Add perturbations
            x = np.linspace(0, 2*np.pi, self.nx)
            y = np.linspace(0, 2*np.pi, self.ny)
            X, Y = np.meshgrid(x, y)
            
            for k in range(self.nz):
                u[:, :, k, t] += 0.5 * np.sin(X) * np.cos(Y)
                v[:, :, k, t] += 0.5 * np.cos(X) * np.sin(Y)
        
        return u, v, w
    
    def generate_pressure_field(self, base_pressure=101325):
        """
        Generate synthetic pressure field
        
        Args:
            base_pressure: Sea level pressure (Pa)
            
        Returns:
            Pressure array (nx, ny, nz, time_steps)
        """
        P = np.zeros((self.nx, self.ny, self.nz, self.time_steps))
        
        for k in range(self.nz):
            # Hydrostatic profile
            height = k * 100
            P[:, :, k, :] = base_pressure * (1 - 0.0065 * height / 288.15) ** 5.255
        
        return P
    
    def get_collocation_points(self, n_points=5000):
        """
        Generate random collocation points for physics loss
        
        Args:
            n_points: Number of collocation points
            
        Returns:
            Collocation point coordinates (n_points, 4)
        """
        x = np.random.uniform(0, self.domain_size, n_points)
        y = np.random.uniform(0, self.domain_size, n_points)
        z = np.random.uniform(0, 5000, n_points)
        t = np.random.uniform(0, self.time_steps * 3600, n_points)  # in seconds
        
        return np.column_stack([x, y, z, t])


def create_training_dataset(n_samples=10000, normalizer=None):
    """
    Create synthetic training dataset for PINN
    
    Args:
        n_samples: Number of training samples
        normalizer: GeoNormalizer instance
        
    Returns:
        Normalized training coordinates and target values
    """
    generator = SyntheticDataGenerator()
    
    if normalizer is None:
        normalizer = GeoNormalizer()
    
    # Generate synthetic fields
    T = generator.generate_temperature_field()
    u, v, w = generator.generate_velocity_field()
    P = generator.generate_pressure_field()
    
    # Randomly sample points
    indices = np.random.choice(
        generator.nx * generator.ny * generator.nz * generator.time_steps,
        n_samples,
        replace=False
    )
    
    idx_x = indices % generator.nx
    idx_y = (indices // generator.nx) % generator.ny
    idx_z = (indices // (generator.nx * generator.ny)) % generator.nz
    idx_t = indices // (generator.nx * generator.ny * generator.nz)
    
    # Extract coordinates and values
    x_vals = idx_x * generator.resolution
    y_vals = idx_y * generator.resolution
    z_vals = idx_z * 100
    t_vals = idx_t * 3600  # 1 hour intervals
    
    u_vals = u[idx_x, idx_y, idx_z, idx_t]
    v_vals = v[idx_x, idx_y, idx_z, idx_t]
    w_vals = w[idx_x, idx_y, idx_z, idx_t]
    p_vals = P[idx_x, idx_y, idx_z, idx_t]
    t_temp_vals = T[idx_x, idx_y, idx_z, idx_t]
    
    # Normalize coordinates
    x_norm, y_norm, z_norm, t_norm = normalizer.normalize_coords(
        x_vals, y_vals, z_vals, t_vals
    )
    
    # Create coordinate matrix
    X_train = np.column_stack([x_norm, y_norm, z_norm, t_norm])
    
    # Create target matrix (u, v, w, p, T)
    y_train = np.column_stack([u_vals, v_vals, w_vals, p_vals, t_temp_vals])
    
    # Normalize outputs to [-1, 1]
    y_train = 2 * (y_train - y_train.min(axis=0)) / (y_train.max(axis=0) - y_train.min(axis=0)) - 1
    
    return X_train, y_train, normalizer
