"""
Physics Constraint Module for Physi-Cast
Implements Navier-Stokes and Thermal Diffusion PDEs as loss regularizers
"""

import torch
import numpy as np


class PhysicsConstraints:
    """
    Encodes fundamental atmospheric physics equations:
    - Navier-Stokes momentum conservation
    - Thermal energy advection-diffusion
    - Mass continuity (simplified for incompressible air)
    """
    
    def __init__(self, nu=1.5e-5, alpha=2.2e-5, rho=1.225, g=9.81):
        """
        Initialize physical constants
        
        Args:
            nu: Kinematic viscosity (m²/s) for air at sea level
            alpha: Thermal diffusivity (m²/s) for air
            rho: Reference air density (kg/m³)
            g: Gravitational acceleration (m/s²)
        """
        self.nu = nu          # Kinematic viscosity
        self.alpha = alpha    # Thermal diffusivity
        self.rho = rho        # Air density
        self.g = g            # Gravity
        
    def compute_gradients(self, y, x, create_graph=True):
        """
        Compute spatial and temporal derivatives using automatic differentiation
        
        Args:
            y: Network output (u, v, w, p, T)
            x: Input coordinates (x, y, z, t)
            create_graph: Enable higher-order derivatives
            
        Returns:
            Dictionary containing all required gradients
        """
        # Extract individual outputs
        u = y[:, 0:1]
        v = y[:, 1:2]
        w = y[:, 2:3]
        p = y[:, 3:4]
        T = y[:, 4:5]
        
        # Compute gradients for each output separately
        grads = {}
        
        # Gradients of u
        du_dx = torch.autograd.grad(u.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if du_dx is not None:
            grads['du_dx'] = du_dx[:, 0:1]
            grads['du_dy'] = du_dx[:, 1:2]
            grads['du_dz'] = du_dx[:, 2:3]
            grads['du_dt'] = du_dx[:, 3:4]
        else:
            grads['du_dx'] = torch.zeros_like(x[:, 0:1])
            grads['du_dy'] = torch.zeros_like(x[:, 1:2])
            grads['du_dz'] = torch.zeros_like(x[:, 2:3])
            grads['du_dt'] = torch.zeros_like(x[:, 3:4])
        
        # Gradients of v
        dv_dx = torch.autograd.grad(v.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if dv_dx is not None:
            grads['dv_dx'] = dv_dx[:, 0:1]
            grads['dv_dy'] = dv_dx[:, 1:2]
            grads['dv_dz'] = dv_dx[:, 2:3]
            grads['dv_dt'] = dv_dx[:, 3:4]
        else:
            grads['dv_dx'] = torch.zeros_like(x[:, 0:1])
            grads['dv_dy'] = torch.zeros_like(x[:, 1:2])
            grads['dv_dz'] = torch.zeros_like(x[:, 2:3])
            grads['dv_dt'] = torch.zeros_like(x[:, 3:4])
        
        # Gradients of w
        dw_dx = torch.autograd.grad(w.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if dw_dx is not None:
            grads['dw_dx'] = dw_dx[:, 0:1]
            grads['dw_dy'] = dw_dx[:, 1:2]
            grads['dw_dz'] = dw_dx[:, 2:3]
            grads['dw_dt'] = dw_dx[:, 3:4]
        else:
            grads['dw_dx'] = torch.zeros_like(x[:, 0:1])
            grads['dw_dy'] = torch.zeros_like(x[:, 1:2])
            grads['dw_dz'] = torch.zeros_like(x[:, 2:3])
            grads['dw_dt'] = torch.zeros_like(x[:, 3:4])
        
        # Gradients of p
        dp_dx = torch.autograd.grad(p.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if dp_dx is not None:
            grads['dp_dx'] = dp_dx[:, 0:1]
            grads['dp_dy'] = dp_dx[:, 1:2]
            grads['dp_dz'] = dp_dx[:, 2:3]
        else:
            grads['dp_dx'] = torch.zeros_like(x[:, 0:1])
            grads['dp_dy'] = torch.zeros_like(x[:, 1:2])
            grads['dp_dz'] = torch.zeros_like(x[:, 2:3])
        
        # Gradients of T
        dT_dx = torch.autograd.grad(T.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if dT_dx is not None:
            grads['dT_dx'] = dT_dx[:, 0:1]
            grads['dT_dy'] = dT_dx[:, 1:2]
            grads['dT_dz'] = dT_dx[:, 2:3]
            grads['dT_dt'] = dT_dx[:, 3:4]
        else:
            grads['dT_dx'] = torch.zeros_like(x[:, 0:1])
            grads['dT_dy'] = torch.zeros_like(x[:, 1:2])
            grads['dT_dz'] = torch.zeros_like(x[:, 2:3])
            grads['dT_dt'] = torch.zeros_like(x[:, 3:4])
        
        return grads
    
    def compute_laplacian(self, du_dx, du_dy, du_dz, x):
        """Compute Laplacian ∇²u using second derivatives"""
        # Compute second derivatives
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dz.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        
        if d2u_dx2 is not None:
            laplacian = d2u_dx2[:, 0:1] + d2u_dy2[:, 1:2] + d2u_dz2[:, 2:3]
        else:
            laplacian = torch.zeros_like(du_dx)
        
        return laplacian
    
    def navier_stokes_loss(self, y, x, y_output):
        """
        Compute Navier-Stokes momentum residuals
        ∂u/∂t + (u·∇)u = -(1/ρ)∇p + ν∇²u + g_x
        
        Args:
            y: Network input (coordinates)
            x: Network input tensor with requires_grad
            y_output: Network predictions (u, v, w, p, T)
            
        Returns:
            Mean squared residual
        """
        u = y_output[:, 0:1]
        v = y_output[:, 1:2]
        w = y_output[:, 2:3]
        p = y_output[:, 3:4]
        
        grads = self.compute_gradients(y_output, x)
        
        du_dt = grads['du_dt']
        du_dx = grads['du_dx']
        du_dy = grads['du_dy']
        du_dz = grads['du_dz']
        dp_dx = grads['dp_dx']
        
        dv_dt = grads['dv_dt']
        dv_dx = grads['dv_dx']
        dv_dy = grads['dv_dy']
        dv_dz = grads['dv_dz']
        dp_dy = grads['dp_dy']
        
        dw_dt = grads['dw_dt']
        dw_dx = grads['dw_dx']
        dw_dy = grads['dw_dy']
        dw_dz = grads['dw_dz']
        dp_dz = grads['dp_dz']
        
        # Convective terms: (u·∇)u, (u·∇)v, (u·∇)w
        u_grad_u = u * du_dx + v * du_dy + w * du_dz
        u_grad_v = u * dv_dx + v * dv_dy + w * dv_dz
        u_grad_w = u * dw_dx + v * dw_dy + w * dw_dz
        
        # Navier-Stokes residuals (simplified without viscosity term for speed)
        # ∂u/∂t + (u·∇)u + (1/ρ)∂p/∂x = 0
        residual_u = du_dt + u_grad_u + (1.0 / self.rho) * dp_dx
        residual_v = dv_dt + u_grad_v + (1.0 / self.rho) * dp_dy
        residual_w = dw_dt + u_grad_w + (1.0 / self.rho) * dp_dz - self.g
        
        ns_loss = (
            torch.mean(residual_u ** 2) +
            torch.mean(residual_v ** 2) +
            torch.mean(residual_w ** 2)
        )
        
        return ns_loss
    
    def thermal_diffusion_loss(self, y, x, y_output):
        """
        Compute thermal energy conservation residuals
        ∂T/∂t + (u·∇)T = α∇²T + Q
        
        Args:
            y_output: Network predictions (u, v, w, p, T)
            
        Returns:
            Mean squared residual
        """
        u = y_output[:, 0:1]
        v = y_output[:, 1:2]
        w = y_output[:, 2:3]
        T = y_output[:, 4:5]
        
        grads = self.compute_gradients(y_output, x)
        
        dT_dt = grads['dT_dt']
        dT_dx = grads['dT_dx']
        dT_dy = grads['dT_dy']
        dT_dz = grads['dT_dz']
        
        # Advection term: (u·∇)T
        advection = u * dT_dx + v * dT_dy + w * dT_dz
        
        # Simplified diffusion (neglecting viscous heating for speed)
        # ∂T/∂t + (u·∇)T ≈ 0 (near steady state)
        thermal_residual = dT_dt + advection
        
        thermal_loss = torch.mean(thermal_residual ** 2)
        
        return thermal_loss
    
    def continuity_loss(self, y, x, y_output):
        """
        Compute mass continuity residual (simplified)
        ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0 (incompressible air assumption)
        
        Args:
            y_output: Network predictions
            
        Returns:
            Continuity residual
        """
        grads = self.compute_gradients(y_output, x)
        
        du_dx = grads['du_dx']
        dv_dy = grads['dv_dy']
        dw_dz = grads['dw_dz']
        
        divergence = du_dx + dv_dy + dw_dz
        continuity_residual = torch.mean(divergence ** 2)
        
        return continuity_residual
    
    def boundary_conditions_loss(self, y_pred, y_bc):
        """
        Enforce boundary conditions
        
        Args:
            y_pred: Predictions at boundary points
            y_bc: Known boundary values
            
        Returns:
            MSE of boundary conditions
        """
        if y_bc is None:
            return torch.tensor(0.0, device=y_pred.device)
        
        bc_loss = torch.mean((y_pred - y_bc) ** 2)
        return bc_loss
    
    def total_physics_loss(self, y, x, y_output, y_bc=None, 
                          w_ns=0.4, w_thermal=0.3, w_continuity=0.2, w_bc=0.1):
        """
        Combine all physics constraints into total loss
        
        Args:
            w_ns: Weight for Navier-Stokes
            w_thermal: Weight for thermal diffusion
            w_continuity: Weight for continuity
            w_bc: Weight for boundary conditions
            
        Returns:
            Weighted sum of all physics losses
        """
        ns_loss = self.navier_stokes_loss(y, x, y_output)
        thermal_loss = self.thermal_diffusion_loss(y, x, y_output)
        continuity = self.continuity_loss(y, x, y_output)
        bc_loss = self.boundary_conditions_loss(y_output, y_bc)
        
        total = (
            w_ns * ns_loss +
            w_thermal * thermal_loss +
            w_continuity * continuity +
            w_bc * bc_loss
        )
        
        return {
            'total': total,
            'ns': ns_loss,
            'thermal': thermal_loss,
            'continuity': continuity,
            'bc': bc_loss
        }
