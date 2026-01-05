"""
SplineVST: Spline-Based Variance Stabilizing Transform for 2D Images

A data-driven, noise-model-agnostic approach to variance stabilization.

Key Insight:
-----------
A good Variance Stabilizing Transform (VST) converts signal-dependent noise into
approximately additive white Gaussian noise (AWGN). Once in the stabilized domain,
a simple Wiener filter becomes optimal.

Training Approach:
-----------------
We minimize the dispersion of log-variance across intensity levels, using a Wiener
filter as a proxy for the underlying signal mean. The Wiener filter is a valid proxy
because:
1. It's monotonic with respect to the underlying mean
2. It's the optimal linear estimator under AWGN (which VST aims to achieve)

Inference Pipeline:
------------------
    z = T(x)           # Transform to stabilized domain
    z_hat = Wiener(z)  # Denoise (now AWGN)
    x_hat = T^{-1}(z_hat)  # Transform back

Key Features:
- Monotonic cubic B-spline transform with hard slope bounds
- Self-consistent proxy-based training (no ground truth needed)
- Compatible with any Gaussian denoiser (Wiener, DRUNet, BM3D, etc.)

Author: Oankar
Date: January 2026
"""

import copy
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# B-SPLINE UTILITIES
# =============================================================================

@torch.jit.script
def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Compute B-spline basis functions using Cox-de Boor recursion.

    Args:
        x: Input values [N, D] where N is batch, D is dimension
        grid: Knot vector [1, num_knots + 2*k]
        k: Spline degree (3 for cubic)

    Returns:
        Basis functions [N, D, num_coefficients]
    """
    x = x.unsqueeze(2)
    grid = grid.unsqueeze(0)

    # Initialize order-0 basis (piecewise constant)
    B = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()

    # Cox-de Boor recursion to build higher-order basis
    for d in range(1, k + 1):
        g0 = grid[:, :, :-(d + 1)]
        gd = grid[:, :, d:-1]
        g1 = grid[:, :, 1:(-d)]
        gdp1 = grid[:, :, d + 1:]

        denom1 = gd - g0 + 1e-8
        denom2 = gdp1 - g1 + 1e-8

        B = ((x - g0) / denom1) * B[:, :, :-1] + ((gdp1 - x) / denom2) * B[:, :, 1:]

    return torch.nan_to_num(B).float()


def evaluate_spline(
    x: torch.Tensor, grid: torch.Tensor, coefficients: torch.Tensor, k: int = 3
) -> torch.Tensor:
    """Evaluate spline at points x given grid and coefficients."""
    basis = bspline_basis(x, grid, k)
    return torch.einsum("ndc,doc->ndo", basis, coefficients).squeeze(-1)


# =============================================================================
# MONOTONIC SPLINE TRANSFORM
# =============================================================================

class MonotonicSpline(nn.Module):
    """
    Monotonic cubic B-spline transform with guaranteed invertibility.

    Monotonicity is enforced by parameterizing coefficients as cumulative sums
    of constrained positive deltas (using sigmoid activation with min/max bounds).

    Args:
        num_knots: Number of internal knot points
        degree: Spline degree (default 3 for cubic)
        min_slope: Minimum allowed slope (ensures strict monotonicity)
        max_slope: Maximum allowed slope (prevents instability)
    """

    def __init__(
        self,
        num_knots: int = 40,
        degree: int = 3,
        min_slope: float = 0.5,
        max_slope: float = 3.0,
    ):
        super().__init__()

        self.degree = degree
        self.num_knots = num_knots
        self.num_coefficients = num_knots + degree

        # Create extended knot vector
        internal = torch.linspace(0.0, 1.0, num_knots + 1)
        grid = self._extend_knots(internal, degree)
        self.register_buffer("grid", grid.unsqueeze(0))

        # Slope constraints
        self.grid_spacing = 1.0 / num_knots
        self.min_delta = min_slope * self.grid_spacing
        self.max_delta = max_slope * self.grid_spacing

        # Learnable parameters: initial coefficient + deltas
        self.c0 = nn.Parameter(torch.zeros(1))
        self.raw_delta = nn.Parameter(torch.zeros(self.num_coefficients - 1))

        # Input normalization (set during initialize())
        self.register_buffer("in_shift", torch.tensor(0.0))
        self.register_buffer("in_scale", torch.tensor(1.0))

        # Output normalization (set during calibrate())
        self.register_buffer("out_shift", torch.tensor(0.0))
        self.register_buffer("out_scale", torch.tensor(1.0))

        self._init_sqrt_like()

    @staticmethod
    def _extend_knots(knots: torch.Tensor, k: int) -> torch.Tensor:
        """Extend knot vector for B-spline evaluation at boundaries."""
        h = knots[-1] - knots[-2]
        extended = knots
        for _ in range(k):
            extended = torch.cat([extended[:1] - h, extended, extended[-1:] + h])
        return extended

    def _init_sqrt_like(self):
        """Initialize spline to approximate sqrt transform (good for Poisson-like noise)."""
        with torch.no_grad():
            x = torch.linspace(0, 1, self.num_coefficients)
            y = torch.sqrt(x.clamp_min(0.001))
            y = (y - y.min()) / (y.max() - y.min())

            deltas = y[1:] - y[:-1]
            deltas = deltas.clamp(self.min_delta + 1e-6, self.max_delta - 1e-6)

            # Convert to raw (pre-sigmoid) parameterization
            normalized = (deltas - self.min_delta) / (self.max_delta - self.min_delta)
            normalized = normalized.clamp(0.01, 0.99)
            raw = torch.log(normalized / (1 - normalized))

            self.c0.fill_(0.0)
            self.raw_delta.copy_(raw)

    @property
    def coefficients(self) -> torch.Tensor:
        """Get monotonic coefficients from raw parameters."""
        deltas = self.min_delta + (self.max_delta - self.min_delta) * torch.sigmoid(self.raw_delta)
        coef = torch.cat([self.c0, deltas]).cumsum(dim=0)
        return coef.view(1, 1, -1)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform without output calibration."""
        x_norm = (x - self.in_shift) / (self.in_scale + 1e-8)
        x_norm = x_norm.clamp(0, 1)

        x_flat = x_norm.flatten().unsqueeze(1)
        y_flat = evaluate_spline(x_flat, self.grid, self.coefficients, self.degree)

        return y_flat.view(x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform with output calibration (standardized output)."""
        y_raw = self.forward_raw(x)
        return (y_raw - self.out_shift) * self.out_scale

    def inverse(self, y: torch.Tensor, num_samples: int = 2048) -> torch.Tensor:
        """
        Inverse transform using lookup table interpolation.

        Since the spline is monotonic, we can invert by:
        1. Building a lookup table y_grid = T(x_grid)
        2. For each y, find bracketing y values and interpolate x
        """
        y_raw = y / (self.out_scale + 1e-8) + self.out_shift

        # Build lookup table
        x_grid = torch.linspace(0, 1, num_samples, device=y.device)
        x_grid_2d = x_grid.unsqueeze(1)
        y_grid = evaluate_spline(x_grid_2d, self.grid, self.coefficients, self.degree).squeeze()

        y_flat = y_raw.flatten()
        y_clamped = y_flat.clamp(y_grid.min(), y_grid.max())

        # Binary search for bracketing indices
        idx = torch.bucketize(y_clamped, y_grid, right=False).clamp(1, num_samples - 1)
        y0, y1 = y_grid[idx - 1], y_grid[idx]
        x0, x1 = x_grid[idx - 1], x_grid[idx]

        # Linear interpolation
        t = (y_clamped - y0) / (y1 - y0 + 1e-8)
        x_norm = x0 + t * (x1 - x0)

        # Denormalize
        x = x_norm * self.in_scale + self.in_shift
        return x.view(y.shape)

    def initialize(self, x: torch.Tensor, max_samples: int = 10_000_000):
        """Set input normalization based on data quantiles."""
        with torch.no_grad():
            x_flat = x.flatten()
            if x_flat.numel() > max_samples:
                idx = torch.randperm(x_flat.numel(), device=x.device)[:max_samples]
                x_flat = x_flat[idx]
            q = torch.tensor([0.001, 0.999], device=x.device)
            lo, hi = torch.quantile(x_flat, q)
            self.in_shift.fill_(lo.item())
            self.in_scale.fill_((hi - lo).clamp_min(1e-6).item())

    def calibrate(self, x: torch.Tensor, max_samples: int = 10_000_000):
        """Set output normalization to standardize the transformed domain."""
        with torch.no_grad():
            if x.numel() > max_samples:
                x_flat = x.flatten()
                idx = torch.randperm(x_flat.numel(), device=x.device)[:max_samples]
                x_subset = x_flat[idx]
            else:
                x_subset = x.flatten()

            y_raw = self.forward_raw(x_subset.view(-1))
            y_min, y_max = y_raw.min(), y_raw.max()
            self.out_shift.fill_(y_min.item())
            self.out_scale.fill_(1.0 / (y_max - y_min + 1e-8).item())


# =============================================================================
# LOCAL WIENER FILTER
# =============================================================================

class LocalWiener2D(nn.Module):
    """
    Local Wiener filter for AWGN denoising.

    Assumes: z = signal + N(0, σ²)

    For each pixel, estimates local mean and variance in a window,
    then applies the Wiener shrinkage formula:
        z_denoised = μ + max(0, σ²_local - σ²_noise) / σ²_local * (z - μ)

    This is optimal (in MSE sense) for AWGN when local statistics are known.

    Args:
        window_size: Size of local window (must be odd)
        eps: Small constant for numerical stability
        pad_mode: Padding mode for boundary handling
    """

    def __init__(self, window_size: int = 7, eps: float = 1e-8, pad_mode: str = "reflect"):
        super().__init__()
        self.window_size = int(window_size)
        assert self.window_size >= 3 and self.window_size % 2 == 1, \
            "window_size must be odd and >= 3"
        self.eps = float(eps)
        self.pad_mode = pad_mode

    def forward(
        self, z: torch.Tensor, noise_std: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply local Wiener filter.

        Args:
            z: Noisy image [B, C, H, W]
            noise_std: Noise standard deviation (scalar or [B] tensor)

        Returns:
            Denoised image [B, C, H, W]
        """
        if z.ndim != 4:
            raise ValueError(f"Expected z as [B,C,H,W], got {tuple(z.shape)}")

        B, C, H, W = z.shape
        k = self.window_size
        pad = k // 2

        # Noise variance broadcast
        if not torch.is_tensor(noise_std):
            noise_var = torch.tensor(float(noise_std) ** 2, device=z.device, dtype=z.dtype)
        else:
            noise_var = noise_std.to(device=z.device, dtype=z.dtype) ** 2

        if noise_var.ndim == 0:
            noise_var = noise_var.view(1, 1, 1, 1)
        elif noise_var.ndim == 1:
            noise_var = noise_var.view(B, 1, 1, 1)

        # Depthwise averaging kernel
        kernel = torch.ones((C, 1, k, k), device=z.device, dtype=z.dtype) / float(k * k)

        z_pad = F.pad(z, (pad, pad, pad, pad), mode=self.pad_mode)

        # Local statistics
        mean = F.conv2d(z_pad, kernel, groups=C)
        mean2 = F.conv2d(z_pad * z_pad, kernel, groups=C)
        var = (mean2 - mean * mean).clamp_min(0.0)

        # Wiener gain: shrink toward mean based on signal-to-noise ratio
        gain = (var - noise_var).clamp_min(0.0) / (var + self.eps)

        return mean + gain * (z - mean)


class WienerDenoiser(nn.Module):
    """Wrapper for LocalWiener2D matching standard denoiser interface."""

    def __init__(self, window_size: int = 7):
        super().__init__()
        self.wiener = LocalWiener2D(window_size=window_size)

    def forward(
        self, z: torch.Tensor, noise_std: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        return self.wiener(z, noise_std)


# =============================================================================
# SPLINE VST 2D MODEL
# =============================================================================

class SplineVST2D(nn.Module):
    """
    Data-driven Variance Stabilizing Transform using monotonic B-splines.

    Training uses a self-consistent proxy approach:
    1. Transform noisy image: z = T(x)
    2. Apply Wiener filter: z_proxy = Wiener(z, σ)
    3. Inverse transform: x_proxy = T⁻¹(z_proxy)
    4. Minimize dispersion of log-variance across intensity bins

    The Wiener filter serves as a monotonic proxy for the underlying mean
    because VST aims to make noise AWGN, where Wiener is optimal.

    Args:
        num_knots: Number of spline knots
        min_slope: Minimum spline slope (monotonicity)
        max_slope: Maximum spline slope (stability)
        blur_sigma: Gaussian blur sigma (fallback proxy)
        self_consistent_proxy: Use self-consistent Wiener proxy
        proxy_update_every: Steps between proxy updates
        proxy_filter: 'wiener' or 'blur'
        wiener_window: Window size for Wiener filter
        proxy_sigma_init: Initial noise std estimate for proxy
        proxy_sigma_ema: EMA momentum for sigma updates
        proxy_sigma_clip: (min, max) bounds for sigma
    """

    def __init__(
        self,
        num_knots: int = 40,
        min_slope: float = 0.5,
        max_slope: float = 3.0,
        blur_sigma: float = 2.5,
        # Self-consistent proxy controls
        self_consistent_proxy: bool = True,
        proxy_update_every: int = 1,
        # Proxy filter settings
        proxy_filter: str = "wiener",
        wiener_window: int = 7,
        # Running sigma for Wiener proxy
        proxy_sigma_init: float = 0.10,
        proxy_sigma_ema: float = 0.95,
        proxy_sigma_clip: Tuple[float, float] = (1e-4, 0.5),
    ):
        super().__init__()

        self.blur_sigma = blur_sigma
        self.spline = MonotonicSpline(
            num_knots=num_knots,
            min_slope=min_slope,
            max_slope=max_slope,
        )

        # Inference-time noise std (set after training)
        self.noise_std = 0.1

        # Proxy settings
        self.self_consistent_proxy = self_consistent_proxy
        self.proxy_update_every = max(1, int(proxy_update_every))

        self.proxy_filter = proxy_filter.lower().strip()
        if self.proxy_filter not in {"wiener", "blur"}:
            raise ValueError("proxy_filter must be 'wiener' or 'blur'")

        self.proxy_wiener = LocalWiener2D(window_size=wiener_window)

        # Running sigma estimate for proxy Wiener
        self.register_buffer("proxy_sigma", torch.tensor(float(proxy_sigma_init)))
        self.proxy_sigma_ema = float(proxy_sigma_ema)
        self.proxy_sigma_clip = (float(proxy_sigma_clip[0]), float(proxy_sigma_clip[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform image to stabilized domain."""
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1)
        z_flat = self.spline(x_flat)
        return z_flat.view(B, H, W, C).permute(0, 3, 1, 2)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from stabilized domain back to original."""
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1)
        x_flat = self.spline.inverse(z_flat)
        return x_flat.view(B, H, W, C).permute(0, 3, 1, 2)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise Gaussian blur."""
        sigma = self.blur_sigma
        ks = max(3, int(6 * sigma + 1) | 1)
        coords = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
        k1d = torch.exp(-coords**2 / (2 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = torch.outer(k1d, k1d)
        B, C, H, W = x.shape

        kernel = k2d.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        pad = ks // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel, groups=C)

    def _proxy_denoise_z(self, z: torch.Tensor) -> torch.Tensor:
        """Denoise in stabilized domain for proxy construction."""
        if self.proxy_filter == "wiener":
            sigma = float(self.proxy_sigma.item())
            return self.proxy_wiener(z, sigma)
        else:
            return self._gaussian_blur(z)

    def _self_consistent_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build self-consistent proxy:
            x → T(x) → Wiener(T(x)) → T⁻¹(...)

        Detached to prevent degenerate solutions.
        """
        with torch.no_grad():
            z = self.forward(x)
            z_proxy = self._proxy_denoise_z(z)
            x_proxy = self.inverse(z_proxy)
        return x_proxy

    def _compute_loss_1d(
        self,
        z: torch.Tensor,
        proxy: torch.Tensor,
        num_bins: int = 28,
        bandwidth: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute variance dispersion loss for flattened 1D data.

        Args:
            z: Transformed values (flat)
            proxy: Proxy values (flat) - used to order/bin the data
            num_bins: Number of intensity bins
            bandwidth: Kernel bandwidth for soft binning

        Returns:
            (dispersion_loss, estimated_noise_std)
        """
        n = proxy.shape[0]

        # Convert proxy values to ranks, then to standard normal quantiles
        order = torch.argsort(proxy)
        ranks = torch.argsort(order).float()
        u = (ranks + 0.5) / n
        z_proxy = torch.erfinv(2 * u.clamp(1e-6, 1 - 1e-6) - 1) * math.sqrt(2)

        # Create bin centers in standard normal space
        q_steps = torch.linspace(0.01, 0.99, num_bins, device=z.device)
        bin_centers = torch.erfinv(2 * q_steps - 1) * math.sqrt(2)

        # Soft binning using Gaussian kernel
        sigma = bandwidth * (bin_centers[1] - bin_centers[0])
        dists = (z_proxy.unsqueeze(-1) - bin_centers.unsqueeze(0)) / sigma
        weights = torch.exp(-0.5 * dists**2) + 1e-12
        weights = weights / weights.sum(dim=-1, keepdim=True)

        weight_sums = weights.sum(dim=0)
        valid = weight_sums > 1.0

        # Compute weighted mean and variance per bin
        z_expanded = z.unsqueeze(-1)
        means = (weights * z_expanded).sum(0) / (weight_sums + 1e-12)
        variances = (weights * (z_expanded - means.unsqueeze(0)) ** 2).sum(0) / (weight_sums + 1e-12)
        variances = variances.clamp(min=1e-8)

        # Loss: dispersion of log-variance (want it constant = 0 dispersion)
        log_var = torch.log(variances[valid])
        dispersion = log_var.var()

        # Estimate noise std from mean variance
        noise_std = torch.sqrt(variances[valid].mean())

        return dispersion, noise_std

    def _compute_loss(
        self,
        z: torch.Tensor,
        proxy: torch.Tensor,
        num_bins: int = 32,
        bandwidth: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for image tensors (flattens internally)."""
        z_flat = z.flatten()
        proxy_flat = proxy.flatten()
        return self._compute_loss_1d(z_flat, proxy_flat, num_bins, bandwidth)

    def fit(
        self,
        images: torch.Tensor,
        num_steps: int = 2000,
        lr: float = 1e-3,
        batch_size: int = 8,
        pixels_per_batch: int = 5_000_000,
        verbose: bool = True,
        # Stabilization options
        loss_ema_beta: float = 0.98,
        save_best: bool = True,
        ema_params: bool = True,
        ema_decay: float = 0.999,
        min_improve: float = 1e-4,
        warmup_steps: int = 0,
    ) -> "SplineVST2D":
        """
        Train the VST on a set of noisy images.

        Args:
            images: Training images [N, C, H, W]
            num_steps: Number of optimization steps
            lr: Learning rate
            batch_size: Images per batch
            pixels_per_batch: Max pixels for loss computation
            verbose: Print progress
            loss_ema_beta: EMA factor for loss tracking
            save_best: Save and restore best checkpoint
            ema_params: Use EMA for parameters
            ema_decay: EMA decay for parameters
            min_improve: Minimum improvement to update best
            warmup_steps: LR warmup steps

        Returns:
            self (fitted model)
        """
        device = images.device
        self.to(device)

        # Initialize input normalization from data
        self.spline.initialize(images)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, num_steps - warmup_steps), eta_min=lr / 50
        )

        n_images = images.shape[0]

        if verbose:
            print("Training SplineVST...")
            print(f"  Steps: {num_steps}, lr={lr}, warmup={warmup_steps}")
            print(f"  Proxy: self-consistent ({self.proxy_filter}), update_every={self.proxy_update_every}")

        # Tracking state
        best_state = None
        best_loss_ema = float("inf")
        loss_ema = None

        # EMA parameters shadow copy
        ema_shadow = None
        if ema_params:
            ema_shadow = {k: v.detach().clone() for k, v in self.state_dict().items()}

        # Cached batch for proxy consistency
        batch_images_cache = None
        batch_proxy_cache = None

        for step in range(num_steps):
            optimizer.zero_grad()

            # Refresh batch and proxy together
            if (batch_images_cache is None) or (step % self.proxy_update_every == 0):
                idx = torch.randperm(n_images, device=device)[:batch_size]
                batch_images_cache = images[idx]
                if self.self_consistent_proxy:
                    batch_proxy_cache = self._self_consistent_proxy(batch_images_cache)
                else:
                    with torch.no_grad():
                        batch_proxy_cache = self._gaussian_blur(batch_images_cache)

            batch_images = batch_images_cache
            batch_proxy = batch_proxy_cache

            # Subsample pixels if needed
            total_pixels = batch_images.numel()
            if total_pixels > pixels_per_batch:
                pix_idx = torch.randperm(total_pixels, device=device)[:pixels_per_batch]
                flat_images = batch_images.flatten()[pix_idx]
                flat_proxy = batch_proxy.flatten()[pix_idx]
            else:
                flat_images = batch_images.flatten()
                flat_proxy = batch_proxy.flatten()

            # Compute loss in raw (uncalibrated) z-space
            z = self.spline.forward_raw(flat_images)
            proxy_z = self.spline.forward_raw(flat_proxy)

            loss, noise_std = self._compute_loss_1d(z, proxy_z, num_bins=24, bandwidth=1.3)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            # LR schedule: warmup then cosine
            if warmup_steps > 0 and step < warmup_steps:
                warm_lr = lr * float(step + 1) / float(warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr
            else:
                scheduler.step()

            # Update proxy sigma estimate
            with torch.no_grad():
                beta = self.proxy_sigma_ema
                self.proxy_sigma.mul_(beta).add_(noise_std.detach() * (1.0 - beta))
                lo, hi = self.proxy_sigma_clip
                self.proxy_sigma.clamp_(lo, hi)

            # Update loss EMA
            l = float(loss.item())
            if loss_ema is None:
                loss_ema = l
            else:
                loss_ema = loss_ema_beta * loss_ema + (1.0 - loss_ema_beta) * l

            # Update parameter EMA
            if ema_params:
                with torch.no_grad():
                    sd = self.state_dict()
                    for k, v in sd.items():
                        ema_shadow[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0 - ema_decay))

            # Save best checkpoint
            if save_best and (loss_ema + min_improve) < best_loss_ema:
                best_loss_ema = loss_ema
                best_state = copy.deepcopy(self.state_dict())

            if verbose and (step + 1) % 500 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  Step {step+1}: loss={l:.4f}, loss_ema={loss_ema:.4f}, "
                      f"sigma_hat={noise_std.item():.4f}, proxy_sigma={float(self.proxy_sigma.item()):.4f}, lr={cur_lr:.2e}")

        # Restore best or EMA parameters
        if save_best and best_state is not None:
            self.load_state_dict(best_state)
        elif ema_params and ema_shadow is not None:
            self.load_state_dict(ema_shadow)

        # Calibrate output normalization
        self.spline.calibrate(images)

        # Estimate final noise std in calibrated domain
        with torch.no_grad():
            subset_idx = torch.randperm(n_images, device=device)[: min(50, n_images)]
            z_subset = self.forward(images[subset_idx])
            proxy_subset = self._gaussian_blur(z_subset)
            _, noise_std = self._compute_loss(z_subset, proxy_subset)
            self.noise_std = noise_std.item()

        if verbose:
            print(f"  Best loss_ema: {best_loss_ema:.4f}")
            print(f"  Final noise std (calibrated z-domain): {self.noise_std:.4f}")
            print("  Done!")

        return self

    def estimate_noise_std(self, images: torch.Tensor) -> float:
        """Estimate noise std in the stabilized domain."""
        with torch.no_grad():
            z = self.forward(images)
            proxy = self._gaussian_blur(z)
            _, noise_std = self._compute_loss(z, proxy)
            return noise_std.item()


# =============================================================================
# DENOISING UTILITIES
# =============================================================================

def denoise_with_splinevst(
    noisy: torch.Tensor,
    vst: SplineVST2D,
    denoiser: nn.Module,
    noise_std: Optional[float] = None,
) -> torch.Tensor:
    """
    Denoise using SplineVST + any Gaussian denoiser.

    Pipeline:
        z = T(noisy)
        z_denoised = denoiser(z, sigma)
        clean = T^{-1}(z_denoised)

    Args:
        noisy: Noisy images [B, C, H, W]
        vst: Trained SplineVST2D model
        denoiser: Gaussian denoiser (e.g., DRUNet, WienerDenoiser)
        noise_std: Noise std in stabilized domain (estimated if None)

    Returns:
        Denoised images [B, C, H, W]
    """
    vst.eval()
    with torch.no_grad():
        z = vst.forward(noisy)
        if noise_std is None:
            noise_std = vst.estimate_noise_std(noisy)
        z_denoised = denoiser(z, noise_std)
        clean = vst.inverse(z_denoised)
    return clean


# =============================================================================
# IMAGE LOADING UTILITIES
# =============================================================================

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load and normalize image to [0, 1] float32."""
    try:
        import imageio.v3 as iio
        arr = iio.imread(path)
    except ImportError:
        from PIL import Image
        arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]  # Take first channel

    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        return arr.astype(np.float32) / max(arr.max(), 1.0)


def load_dataset(
    root: Union[str, Path],
    sample_ids: list,
    image_size: int = 512,
    images_per_sample: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load paired noisy/GT data from FMD-style directory structure."""
    root = Path(root)
    noisy_list, gt_list = [], []

    for sid in sample_ids:
        gt_path = root / 'gt' / str(sid) / 'avg50.png'
        if not gt_path.exists():
            continue

        gt = load_image(gt_path)[:image_size, :image_size]

        raw_dir = root / 'raw' / str(sid)
        raw_files = sorted(raw_dir.rglob('*.png'))[:images_per_sample]

        for raw_path in raw_files:
            noisy = load_image(raw_path)[:image_size, :image_size]
            noisy_list.append(noisy)
            gt_list.append(gt)

    noisy = torch.from_numpy(np.stack(noisy_list)).unsqueeze(1)
    gt = torch.from_numpy(np.stack(gt_list)).unsqueeze(1)
    return noisy, gt


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PSNR in dB."""
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


# =============================================================================
# GENERALIZED ANSCOMBE TRANSFORM (ORACLE BASELINE)
# =============================================================================

def gat_forward(x: torch.Tensor, a: float) -> torch.Tensor:
    """Generalized Anscombe Transform for Poisson noise."""
    return (2/a) * torch.sqrt(a * x.clamp_min(0) + 0.375 * a**2)


def gat_inverse(z: torch.Tensor, a: float) -> torch.Tensor:
    """Inverse GAT."""
    return ((z * a / 2)**2 - 0.375 * a**2) / a


# =============================================================================
# TEST SCRIPT
# =============================================================================

def test_splinevst(data_root: str, device: str = 'cuda') -> dict:
    """Test SplineVST on confocal fish data."""
    try:
        from deepinv.models import DRUNet
    except ImportError:
        print("deepinv not installed. Install with: pip install deepinv")
        return {}

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_ids = [str(i) for i in range(1, 19)]
    test_ids = ['19']

    train_noisy, train_gt = load_dataset(data_root, train_ids, images_per_sample=50)
    test_noisy, test_gt = load_dataset(data_root, test_ids, images_per_sample=50)

    train_noisy = train_noisy.to(device)
    train_gt = train_gt.to(device)
    test_noisy = test_noisy.to(device)
    test_gt = test_gt.to(device)

    print(f"  Train: {train_noisy.shape[0]} images")
    print(f"  Test: {test_noisy.shape[0]} images")

    # Known noise parameters (for oracle comparison)
    a_true = 0.0943  # Poisson parameter
    target_sigma = 25/255  # Target noise std for DRUNet

    # Load denoiser
    print("\nLoading DRUNet...")
    drunet = DRUNet(in_channels=1, out_channels=1,
                    pretrained='download', device=device)
    drunet.eval()

    # =========================================================================
    # SplineVST (blind, no noise model required)
    # =========================================================================
    print("\n" + "="*60)
    print("SPLINEVST (Blind)")
    print("="*60)

    vst = SplineVST2D(
        num_knots=40,
        min_slope=0.75,
        max_slope=2.0,
        blur_sigma=50.0,
        self_consistent_proxy=True,
        proxy_update_every=1,
        proxy_filter="wiener",
        wiener_window=9,
        proxy_sigma_init=0.10,
        proxy_sigma_ema=0.99,
    )

    vst.fit(train_noisy, num_steps=1500, verbose=True)

    if hasattr(vst, "proxy_sigma"):
        print(f"  Training proxy_sigma (EMA): {float(vst.proxy_sigma.item()):.4f}")

    # Denoise test images
    with torch.no_grad():
        z = vst.forward(test_noisy)
        noise_std = vst.estimate_noise_std(test_noisy)
        z_denoised = drunet(z, noise_std)
        splinevst_result = vst.inverse(z_denoised)

    psnr_splinevst = psnr(splinevst_result, test_gt)
    print(f"\n  Test noise std (calibrated z-domain): {noise_std:.4f}")
    print(f"  PSNR: {psnr_splinevst:.2f} dB")

    # =========================================================================
    # GAT (Oracle - knows true noise parameter)
    # =========================================================================
    print("\n" + "="*60)
    print("GAT (Oracle)")
    print("="*60)

    with torch.no_grad():
        z_gat = gat_forward(test_noisy, a_true) * target_sigma
        z_gat_denoised = drunet(z_gat, target_sigma)
        gat_result = gat_inverse(z_gat_denoised / target_sigma, a_true)
        gat_result = gat_result.clamp(0, 1)

    psnr_gat = psnr(gat_result, test_gt)
    print(f"  PSNR: {psnr_gat:.2f} dB")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
    Method          PSNR (dB)    Notes
    ------------------------------------------------
    SplineVST       {psnr_splinevst:.2f}         Blind, no noise model
    GAT             {psnr_gat:.2f}         Oracle (knows a={a_true})

    Gap: {psnr_gat - psnr_splinevst:.2f} dB

    SplineVST achieves {100 - 100*(psnr_gat - psnr_splinevst)/psnr_gat:.1f}% of oracle performance
    without knowing the noise model!
    """)

    return {
        'splinevst': psnr_splinevst,
        'gat': psnr_gat,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test SplineVST')
    parser.add_argument('--data-root', type=str, default='/content/confocal_fish',
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    test_splinevst(args.data_root, args.device)
