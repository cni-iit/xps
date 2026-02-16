import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['font.size'] = 18
from scipy.optimize import curve_fit, minimize
from scipy import integrate
from scipy.special import wofz
from scipy.stats import norm
from scipy.signal import savgol_filter
import pandas as pd
import yaml
import json
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Union
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('XPSFit')

def read_xps_csv(file_path):
    metadata = {}
    header_lines = []
    
    # Read file once line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.strip())
            else:
                # First non-# line is the CSV header, stop reading header
                break
    
    # Extract dwell time and number of scans from header
    for line in header_lines:
        if "Dwell Time" in line:
            metadata["dwell_time"] = float(line.split(':', 1)[1].strip())
        if "Number of Scans" in line:
            metadata["n_scans"] = int(line.split(':', 1)[1].strip())
    
    # Now read numeric data
    df = pd.read_csv(file_path, comment='#')
    return df, metadata

# Define lineshape functions
def gaussian(x, amplitude, center, fwhm):
    """Gaussian peak function."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, fwhm):
    """Lorentzian peak function."""
    gamma = fwhm / 2
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def voigt(x, amplitude, center, fwhm_g, fwhm_l):
    """Accurate Voigt profile using the Faddeeva function (scipy.special.wofz)."""
    fwhm_g = max(fwhm_g, 1e-6)
    fwhm_l = max(fwhm_l, 1e-6)
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return amplitude * profile

def doniach_sunjic(x, amplitude, center, fwhm, asymmetry, epsilon=1e-10):
    """
    Corrected Doniach-Sunjic line shape for XPS.
    Asymmetry tail appears on HIGH binding energy side (left in descending XPS plot).

    Parameters:
    -----------
    x : array
        Binding energy (eV) – typically in descending order (high→low BE)
    amplitude : float
        Scaling factor (peak height approximates amplitude * cos(π·α/2) / γ^(1-α) )
    center : float
        Peak position (eV)
    fwhm : float
        Full width at half maximum (eV)
    asymmetry : float
        Asymmetry parameter (0 = symmetric Lorentzian)
        Positive values give tail on HIGH binding energy side.
        Typical range: 0.02–0.2 for metals.
    epsilon : float
        Small number to avoid division by zero.
    """
    gamma = max(fwhm / 2, epsilon)          # HWHM
    alpha = asymmetry

    # Use (center - x) so that for x > center (high BE) the argument is negative.
    # This makes the arctan term negative, producing a tail on the high BE side.
    arg = (center - x) / gamma
    safe_arg = np.where(np.abs(arg) < epsilon, epsilon, arg)

    numerator = np.cos(np.pi * alpha / 2 + (1 - alpha) * np.arctan(safe_arg))
    denominator = ((x - center)**2 + gamma**2) ** ((1 - alpha) / 2)
    denominator = np.where(denominator < epsilon, epsilon, denominator)

    return amplitude * numerator / denominator

def asymmetric_voigt(
    x,
    area,
    center,
    fwhm_g,
    fwhm_l,
    asymmetry=0.0
):
    """
    Physically meaningful asymmetric Voigt.

    Parameters
    ----------
    x : array
    area : float
        Integrated peak area (NOT height!)
    center : float
    fwhm_g : float
        Gaussian FWHM (usually instrumental -> often fixed)
    fwhm_l : float
        Lorentzian FWHM
    asymmetry : float
        0 = symmetric
        typical XPS metal peaks: 0.02 – 0.2

    Returns
    -------
    peak : array
    """

    # --- safety ---
    fwhm_g = max(fwhm_g, 1e-6)
    fwhm_l = max(fwhm_l, 1e-6)

    # convert FWHM -> sigma/gamma
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))

    # asymmetry as linear broadening on high BE side
    gamma = fwhm_l / 2

    gamma_x = gamma * (1 + asymmetry * (x - center))

    # forbid negative widths
    gamma_x = np.clip(gamma_x, gamma * 0.2, gamma * 5)

    z = ((x - center) + 1j * gamma_x) / (sigma * np.sqrt(2))

    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    # normalize by area
    profile /= np.trapezoid(profile, x)

    return area * profile

# Background functions
def linear_background(x, slope, intercept):
    """Linear background."""
    return slope * x + intercept

def shirley_background(x, y, tol=1e-5, max_iter=50, edge_pts=1):
    """
    Calculate iterative Shirley background.
    
    Args:
        x: binding energy array (should be in descending order for XPS)
        y: intensity array
        tol: convergence tolerance
        max_iter: maximum number of iterations
        edge_pts: number of points to average at each edge for baseline estimation
        
    Returns:
        background array
    """
    # Ensure x is in descending order (higher to lower binding energy)
    if x[0] < x[-1]:
        x = x[::-1]
        y = y[::-1]
        reversed = True
    else:
        reversed = False

    # Edge level estimates (average over edge_pts)
    y_highE = np.mean(y[:edge_pts])      # first points (high BE, left side in XPS)
    y_lowE = np.mean(y[-edge_pts:])      # last points (low BE, right side in XPS)

    # Initial background (flat at low-energy side)
    background = np.ones_like(y) * y_lowE

    # Iterative Shirley background calculation
    for _ in range(max_iter):
        # Integrated difference (trapezoidal)
        integral = np.zeros_like(y)
        for i in range(len(y) - 2, -1, -1):  # integrate backward
            integral[i] = integral[i+1] + 0.5 * ( (y[i] - background[i]) +
                                                  (y[i+1] - background[i+1]) ) * (x[i] - x[i+1])

        # Normalize
        if integral[0] != 0:
            integral /= integral[0]

        # Update background using Shirley equation
        new_background = y_lowE + (y_highE - y_lowE) * integral

        # Convergence check
        if np.max(np.abs(new_background - background)) < tol:
            background = new_background
            break

        background = new_background

    if reversed:
        background = background[::-1]

    return background


def tougaard_background(x, y, B=2866, C=1643, D=1, T=1):
    """
    Calculate Tougaard background for XPS.
    
    Args:
        x: binding energy array
        y: intensity array
        B, C, D, T: Tougaard parameters
        
    Returns:
        background array
    """
    # Ensure x is in descending order (higher to lower binding energy)
    if x[0] < x[-1]:
        x = x[::-1]
        y = y[::-1]
        reversed = True
    else:
        reversed = False
    
    # Calculate background
    background = np.zeros_like(y)
    dx = abs(x[1] - x[0])
    
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            # Tougaard cross-section
            E = abs(x[i] - x[j])
            K = B * E / ((C + E**2)**2 + D*E**2)
            
            # Add contribution
            background[i] += K * y[j] * dx
    
    # Scale to match the high binding energy end
    scale = y[0] / background[0] if background[0] > 0 else 1
    background = background * scale
    
    if reversed:
        background = background[::-1]
        
    return background

def calculate_voigt_fwhm(amplitude, center, fwhm_g, fwhm_l, x_range=None, n_points=1000):
    """
    Calculate the FWHM of a Voigt profile numerically.
    
    Parameters:
    amplitude, center, fwhm_g, fwhm_l: Voigt parameters
    x_range: range to evaluate (default: center ± 5*max(fwhm_g, fwhm_l))
    n_points: number of points for evaluation
    
    Returns:
    fwhm: calculated full width at half maximum
    """
    # Create evaluation range if not provided
    if x_range is None:
        width_estimate = max(fwhm_g, fwhm_l) * 5
        x_range = np.linspace(center - width_estimate, center + width_estimate, n_points)
    
    # Evaluate the Voigt function
    y = voigt(x_range, amplitude, center, fwhm_g, fwhm_l)
    
    # Find the maximum value
    max_val = np.max(y)
    half_max = max_val / 2
    
    # Find indices where the function crosses half maximum
    above_half = y > half_max
    left_idx = np.where(above_half)[0][0]
    right_idx = np.where(above_half)[0][-1]
    
    # Interpolate to find exact crossing points
    # Left side
    x_left = np.interp(half_max, y[left_idx-1:left_idx+1][::-1], 
                       x_range[left_idx-1:left_idx+1][::-1])
    # Right side
    x_right = np.interp(half_max, y[right_idx:right_idx+2], 
                        x_range[right_idx:right_idx+2])
    
    return abs(x_right - x_left)

def calculate_doniach_sunjic_widths(amplitude, center, fwhm, asymmetry, x_range=None, n_points=1000):
    """
    Calculate left and right half-widths at half maximum for Doniach-Sunjic function.
    
    Returns:
    left_hwhm: left half-width at half maximum
    right_hwhm: right half-width at half maximum
    """
    # Create evaluation range if not provided
    if x_range is None:
        width_estimate = fwhm * 5
        x_range = np.linspace(center - width_estimate, center + width_estimate, n_points)
    
    # Evaluate the function
    y = doniach_sunjic(x_range, amplitude, center, fwhm, asymmetry)
    
    # Find the maximum value
    max_val = np.max(y)
    half_max = max_val / 2
    
    # Find the peak position
    peak_idx = np.argmax(y)
    
    # Find left and right crossing points
    # Left side
    left_side = y[:peak_idx]
    left_x = x_range[:peak_idx]
    left_cross = left_x[np.where(left_side >= half_max)[0][0]] if np.any(left_side >= half_max) else left_x[0]
    
    # Right side
    right_side = y[peak_idx:]
    right_x = x_range[peak_idx:]
    right_cross = right_x[np.where(right_side >= half_max)[0][-1]] if np.any(right_side >= half_max) else right_x[-1]
    
    left_hwhm = center - left_cross
    right_hwhm = right_cross - center
    
    return left_hwhm, right_hwhm

def calculate_asymmetric_voigt_fwhm(amplitude, center, fwhm_g, fwhm_l, asymmetry, x_range=None, n_points=1000):
    """
    Calculate the FWHM of an asymmetric Voigt profile numerically.
    
    Returns:
    fwhm: calculated full width at half maximum
    left_hwhm: left half-width at half maximum
    right_hwhm: right half-width at half maximum
    """
    # Create evaluation range if not provided
    if x_range is None:
        width_estimate = max(fwhm_g, fwhm_l) * 5
        x_range = np.linspace(center - width_estimate, center + width_estimate, n_points)
    
    # Evaluate the function
    y = asymmetric_voigt(x_range, amplitude, center, fwhm_g, fwhm_l, asymmetry)
    
    # Find the maximum value
    max_val = np.max(y)
    half_max = max_val / 2
    
    # Find the peak position
    peak_idx = np.argmax(y)
    
    # Find left and right crossing points
    # Left side
    left_side = y[:peak_idx]
    left_x = x_range[:peak_idx]
    left_cross = left_x[np.where(left_side >= half_max)[0][0]] if np.any(left_side >= half_max) else left_x[0]
    
    # Right side
    right_side = y[peak_idx:]
    right_x = x_range[peak_idx:]
    right_cross = right_x[np.where(right_side >= half_max)[0][-1]] if np.any(right_side >= half_max) else right_x[-1]
    
    fwhm = right_cross - left_cross
    left_hwhm = center - left_cross
    right_hwhm = right_cross - center
    
    return abs(fwhm), left_hwhm, right_hwhm



@dataclass
class PeakConfig:
    """Configuration for a peak in XPS fitting."""
    peak_type: str  # 'gaussian', 'lorentzian', 'voigt', 'doniach_sunjic', 'asymmetric_voigt'
    peak_name: str
    initial_amplitude: float
    initial_center: float
    initial_fwhm: float
    initial_fwhm_l: Optional[float] = None  # For voigt
    initial_asymmetry: Optional[float] = None  # For asymmetric peaks
    
    # Constraints
    amplitude_bounds: Tuple[float, float] = (0, np.inf)
    center_bounds: Tuple[float, float] = (-np.inf, np.inf)
    fwhm_bounds: Tuple[float, float] = (0.1, 10.0)
    fwhm_l_bounds: Optional[Tuple[float, float]] = None
    asymmetry_bounds: Optional[Tuple[float, float]] = None
    
    # Fixed parameters (if True, the parameter won't be varied during fitting)
    fix_amplitude: bool = False
    fix_center: bool = False
    fix_fwhm: bool = False
    fix_fwhm_l: bool = False
    fix_asymmetry: bool = False

@dataclass
class FitConfig:
    """Configuration for XPS spectrum fitting."""
    # Energy range for fitting
    energy_range: Tuple[float, float]
    
    # Peaks configuration
    peaks: List[PeakConfig]
    
    # Background configuration
    background_type: str  # 'linear', 'shirley', 'tougaard', 'none'
    background_params: Optional[Dict] = None
    
    # Additional fitting options
    max_iterations: int = 1000
    ftol: float = 1e-8
    method: str = 'lm'  # 'lm', 'trf', 'dogbox'
    
    # Sigma type for weighting residuals
    sigma_type: str = 'poisson'  # 'poisson', 'gamma', 'constant'
    sigma_value: float = 30.0  # Used only if sigma_type is 'constant'

class XPSFitter:
    def __init__(self, spectrum=None):
        """Initialize the XPS fitter with an optional spectrum."""
        self.spectrum = spectrum
        self.fit_config = FitConfig(
            energy_range=(0.0, 0.0),
            peaks=[],
            background_type='none',
            background_params={},
            max_iterations=1000,
            ftol=1e-8,
            method='lm',
            sigma_type='poisson',
            sigma_value=30
        )
        self.fit_result = None
        self.background = None
        self.peak_components = []
        self.x_fit = None  # x values used for fitting (may be a subset of spectrum)
        self.y_fit = None  # y values used for fitting
        
    def load_spectrum(self, spectrum):
        """Load a spectrum object."""
        self.spectrum = spectrum
        return self
        
    def load_config_from_file(self, config_path):
        """Load fitting configuration from a YAML or JSON file."""
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'r') as f:
            if ext.lower() == '.yaml' or ext.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif ext.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file extension: {ext}")
        
        return self.load_config_from_dict(config_dict)
    
    def load_config_from_dict(self, config_dict):
        """Load fitting configuration from a dictionary."""
        # Parse energy range
        energy_range = tuple(config_dict.get('energy_range', (None, None)))
        
        # Parse background configuration
        background_type = config_dict.get('background_type', 'none')
        background_params = config_dict.get('background_params', {})
        
        # Parse peaks
        peaks_configs = []
        for peak_dict in config_dict.get('peaks', []):
            peak_config = PeakConfig(
                peak_name=peak_dict.get('name', 'unknown'),
                peak_type=peak_dict.get('type', 'gaussian'),
                initial_amplitude=peak_dict.get('initial_amplitude', 1.0),
                initial_center=peak_dict.get('initial_center', 0.0),
                initial_fwhm=peak_dict.get('initial_fwhm', 1.0),
                initial_fwhm_l=peak_dict.get('initial_fwhm_l'),
                initial_asymmetry=peak_dict.get('initial_asymmetry'),
                
                amplitude_bounds=tuple(peak_dict.get('amplitude_bounds', (0, np.inf))),
                center_bounds=tuple(peak_dict.get('center_bounds', (-np.inf, np.inf))),
                fwhm_bounds=tuple(peak_dict.get('fwhm_bounds', (0.1, 10.0))),
                fwhm_l_bounds=tuple(peak_dict.get('fwhm_l_bounds', (0.1, 10.0))) if peak_dict.get('fwhm_l_bounds') else None,
                asymmetry_bounds=tuple(peak_dict.get('asymmetry_bounds', (0, 1))) if peak_dict.get('asymmetry_bounds') else None,
                
                fix_amplitude=peak_dict.get('fix_amplitude', False),
                fix_center=peak_dict.get('fix_center', False),
                fix_fwhm=peak_dict.get('fix_fwhm', False),
                fix_fwhm_l=peak_dict.get('fix_fwhm_l', False),
                fix_asymmetry=peak_dict.get('fix_asymmetry', False)
            )
            peaks_configs.append(peak_config)
        
        # Create fit configuration
        self.fit_config = FitConfig(
            energy_range=energy_range,
            background_type=background_type,
            background_params=background_params,
            peaks=peaks_configs,
            max_iterations=config_dict.get('max_iterations', 1000),
            ftol=config_dict.get('ftol', 1e-8),
            method=config_dict.get('method', 'lm'),
            sigma_type=config_dict.get('sigma_type', 'poisson'),
        )
        
        return self
    
    def _prepare_data_for_fitting(self):
        """Prepare spectrum data for fitting based on the energy range in config."""
        if self.spectrum is None:
            raise ValueError("No spectrum loaded. Call load_spectrum() first.")
        
        if self.fit_config is None:
            raise ValueError("No fit configuration loaded.")
        
        # Get the binding energy and counts data
        x = np.array(self.spectrum.binding_energy)
        y = np.array(self.spectrum.counts_per_second)
        
        # Sort the data if it's not in descending order (standard for XPS)
        if x[0] < x[-1]:
            sort_idx = np.argsort(x)[::-1]  # Descending order
        else:
            sort_idx = np.arange(len(x))
            
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Apply energy range filter if specified
        emin, emax = self.fit_config.energy_range
        if emin is not None or emax is not None:
            # Adjust for XPS convention (higher binding energy on left)
            if emin is not None and emax is not None and emin > emax:
                emin, emax = emax, emin
                
            mask = np.ones_like(x, dtype=bool)
            if emin is not None:
                mask = mask & (x >= emin)
            if emax is not None:
                mask = mask & (x <= emax)
                
            x = x[mask]
            y = y[mask]
        
        self.x_fit = x
        self.y_fit = y
        
        return x, y
    
    def _compute_background(self, x, y):
        """Compute the background based on the fit configuration."""
        bg_type = self.fit_config.background_type.lower()
        params = self.fit_config.background_params or {}
        
        if bg_type == 'none':
            return np.zeros_like(y)
        elif bg_type == 'linear':
            # Use the first and last points to define linear background
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            intercept = y[0] - slope * x[0]
            return linear_background(x, slope, intercept)
        elif bg_type == 'shirley':
            tol = params.get('tolerance', 1e-5)
            max_iter = params.get('max_iterations', 50)
            return shirley_background(x, y, tol, max_iter)
        elif bg_type == 'tougaard':
            B = params.get('B', 2866)
            C = params.get('C', 1643)
            D = params.get('D', 1)
            T = params.get('T', 1)
            return tougaard_background(x, y, B, C, D, T)
        else:
            raise ValueError(f"Unknown background type: {bg_type}")
    
    def _create_model_function(self):
        """Create the combined model function for fitting."""
        peaks = self.fit_config.peaks
        
        def model_function(x, *params):
            """Combined model of all peaks."""
            y_model = np.zeros_like(x)
            param_idx = 0
            
            for peak in peaks:
                peak_type = peak.peak_type.lower()
                
                if peak_type == 'gaussian':
                    # Parameters: amplitude, center, fwhm
                    n_params = 3
                    amplitude, center, fwhm = params[param_idx:param_idx+n_params]
                    y_model += gaussian(x, amplitude, center, fwhm)
                    
                elif peak_type == 'lorentzian':
                    # Parameters: amplitude, center, fwhm
                    n_params = 3
                    amplitude, center, fwhm = params[param_idx:param_idx+n_params]
                    y_model += lorentzian(x, amplitude, center, fwhm)
                    
                elif peak_type == 'voigt':
                    # Parameters: amplitude, center, fwhm_g, fwhm_l
                    n_params = 4
                    amplitude, center, fwhm_g, fwhm_l = params[param_idx:param_idx+n_params]
                    y_model += voigt(x, amplitude, center, fwhm_g, fwhm_l)
                    
                elif peak_type == 'doniach_sunjic':
                    # Parameters: amplitude, center, fwhm, asymmetry
                    n_params = 4
                    amplitude, center, fwhm, asymmetry = params[param_idx:param_idx+n_params]
                    y_model += doniach_sunjic(x, amplitude, center, fwhm, asymmetry)
                    
                elif peak_type == 'asymmetric_voigt':
                    # Parameters: amplitude, center, fwhm_g, fwhm_l, asymmetry
                    n_params = 5
                    amplitude, center, fwhm_g, fwhm_l, asymmetry = params[param_idx:param_idx+n_params]
                    y_model += asymmetric_voigt(x, amplitude, center, fwhm_g, fwhm_l, asymmetry)
                    
                else:
                    raise ValueError(f"Unknown peak type: {peak_type}")
                    
                param_idx += n_params
                
            return y_model
            
        return model_function
    
    def _prepare_initial_params_and_bounds(self):
        """Prepare initial parameters and bounds for fitting."""
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        fixed_params_mask = []  # True for fixed parameters
        
        for peak in self.fit_config.peaks:
            peak_type = peak.peak_type.lower()
            
            # Add amplitude, center, fwhm parameters for all peak types
            params = [
                peak.initial_amplitude,
                peak.initial_center,
                peak.initial_fwhm
            ]
            
            lower_bounds = [
                peak.amplitude_bounds[0],
                peak.center_bounds[0],
                peak.fwhm_bounds[0]
            ]
            
            upper_bounds = [
                peak.amplitude_bounds[1],
                peak.center_bounds[1],
                peak.fwhm_bounds[1]
            ]
            
            fixed_mask = [
                peak.fix_amplitude,
                peak.fix_center,
                peak.fix_fwhm
            ]
            
            # Add additional parameters for specific peak types
            if peak_type in ['voigt', 'asymmetric_voigt']:
                # Add fwhm_l
                params.append(peak.initial_fwhm_l or peak.initial_fwhm)
                lower_bounds.append((peak.fwhm_l_bounds or peak.fwhm_bounds)[0])
                upper_bounds.append((peak.fwhm_l_bounds or peak.fwhm_bounds)[1])
                fixed_mask.append(peak.fix_fwhm_l)
                
            if peak_type in ['doniach_sunjic', 'asymmetric_voigt']:
                # Add asymmetry
                params.append(peak.initial_asymmetry or 0.1)
                lower_bounds.append((peak.asymmetry_bounds or (0, 1))[0])
                upper_bounds.append((peak.asymmetry_bounds or (0, 1))[1])
                fixed_mask.append(peak.fix_asymmetry)
            
            initial_params.extend(params)
            bounds_lower.extend(lower_bounds)
            bounds_upper.extend(upper_bounds)
            fixed_params_mask.extend(fixed_mask)
            
        # Process fixed parameters
        final_params = []
        final_bounds_lower = []
        final_bounds_upper = []
        fixed_param_values = {}
        
        for i, (param, lower, upper, fixed) in enumerate(zip(
            initial_params, bounds_lower, bounds_upper, fixed_params_mask)):
            
            if fixed:
                fixed_param_values[i] = param
            else:
                final_params.append(param)
                final_bounds_lower.append(lower)
                final_bounds_upper.append(upper)
        
        return (
            np.array(final_params),
            (np.array(final_bounds_lower), np.array(final_bounds_upper)),
            fixed_param_values
        )
    
    def _create_objective_function_with_fixed_params(self, model_func, x, y, fixed_param_values):
        """Create objective function that accounts for fixed parameters."""
        def objective_func(params):
            # Create full parameter list, inserting fixed values
            full_params = np.zeros(len(params) + len(fixed_param_values))
            free_idx = 0
            
            for i in range(len(full_params)):
                if i in fixed_param_values:
                    full_params[i] = fixed_param_values[i]
                else:
                    full_params[i] = params[free_idx]
                    free_idx += 1
            
            # Calculate residuals
            return y - model_func(x, *full_params)
        
        return objective_func
    
    def _extract_peak_components(self, x, params):
        """Extract individual peak components from the fit parameters."""
        components = []
        param_idx = 0
        
        for peak in self.fit_config.peaks:
            peak_type = peak.peak_type.lower()
            
            if peak_type == 'gaussian':
                n_params = 3
                amplitude, center, fwhm = params[param_idx:param_idx+n_params]
                y_component = gaussian(x, amplitude, center, fwhm)
                components.append({
                    'name': peak.peak_name,
                    'type': 'gaussian',
                    'params': {
                        'amplitude': amplitude,
                        'center': center,
                        'fwhm': fwhm
                    },
                    'y_values': y_component
                })
                
            elif peak_type == 'lorentzian':
                n_params = 3
                amplitude, center, fwhm = params[param_idx:param_idx+n_params]
                y_component = lorentzian(x, amplitude, center, fwhm)
                components.append({
                    'name': peak.peak_name,
                    'type': 'lorentzian',
                    'params': {
                        'amplitude': amplitude,
                        'center': center,
                        'fwhm': fwhm
                    },
                    'y_values': y_component
                })
                
            elif peak_type == 'voigt':
                n_params = 4
                amplitude, center, fwhm_g, fwhm_l = params[param_idx:param_idx+n_params]
                y_component = voigt(x, amplitude, center, fwhm_g, fwhm_l)
                components.append({
                    'name': peak.peak_name,
                    'type': 'voigt',
                    'params': {
                        'amplitude': amplitude,
                        'center': center,
                        'fwhm_g': fwhm_g,
                        'fwhm_l': fwhm_l
                    },
                    'y_values': y_component
                })
                
            elif peak_type == 'doniach_sunjic':
                n_params = 4
                amplitude, center, fwhm, asymmetry = params[param_idx:param_idx+n_params]
                y_component = doniach_sunjic(x, amplitude, center, fwhm, asymmetry)
                components.append({
                    'name': peak.peak_name,
                    'type': 'doniach_sunjic',
                    'params': {
                        'amplitude': amplitude,
                        'center': center,
                        'fwhm': fwhm,
                        'asymmetry': asymmetry
                    },
                    'y_values': y_component
                })
                
            elif peak_type == 'asymmetric_voigt':
                n_params = 5
                amplitude, center, fwhm_g, fwhm_l, asymmetry = params[param_idx:param_idx+n_params]
                y_component = asymmetric_voigt(x, amplitude, center, fwhm_g, fwhm_l, asymmetry)
                components.append({
                    'name': peak.peak_name,
                    'type': 'asymmetric_voigt',
                    'params': {
                        'amplitude': amplitude,
                        'center': center,
                        'fwhm_g': fwhm_g,
                        'fwhm_l': fwhm_l,
                        'asymmetry': asymmetry
                    },
                    'y_values': y_component
                })
                
            # components.append({
            #     'name': peak.peak_name,
            # })
            param_idx += n_params
            
        return components
    
    def fit(self, dwell_time=0.096, n_scans=30):
        """Perform the fitting."""
        # Prepare data
        x, y = self._prepare_data_for_fitting()
        
        # Calculate background
        background = self._compute_background(x, y)
        self.background = background
        
        # Subtract background for fitting
        y_no_bg = y - background
        
        # Create model function
        model_func = self._create_model_function()
        
        # Prepare initial parameters and bounds
        initial_params, bounds, fixed_param_values = self._prepare_initial_params_and_bounds()
        logger.info(f"Number of free params: {len(initial_params)}")
        # Estimate sigma for fitting
        # sigma = np.sqrt(np.clip(y_no_bg, 0, None) + 0.1)  # Avoid zero or negative values
        
        # If we have fixed parameters, use a different approach
        if fixed_param_values:
            # Create objective function that handles fixed parameters
            obj_func = self._create_objective_function_with_fixed_params(
                model_func, x, y_no_bg, fixed_param_values)
            
            # Use minimize instead of curve_fit for more flexibility
            # Convert bounds format: from (lower_array, upper_array) to list of tuples
            bounds_list = list(zip(bounds[0], bounds[1]))
            
            res = minimize(
                lambda p: np.sum(obj_func(p)**2),
                initial_params,
                bounds=bounds_list,
                method='L-BFGS-B',
                options={'maxiter': self.fit_config.max_iterations}
            )
            
            # Create full parameter list
            full_params = np.zeros(len(initial_params) + len(fixed_param_values))
            free_idx = 0
            
            for i in range(len(full_params)):
                if i in fixed_param_values:
                    full_params[i] = fixed_param_values[i]
                else:
                    full_params[i] = res.x[free_idx]
                    free_idx += 1
                    
            params = full_params
            
        else:
            # Use curve_fit
            params, pcov = curve_fit(
                model_func,
                x, y_no_bg,
                p0=initial_params,
                bounds=bounds,
                method=self.fit_config.method,
                maxfev=self.fit_config.max_iterations,
                ftol=self.fit_config.ftol
            )
            
            # Calculate parameter errors from covariance matrix
            perr = np.sqrt(np.diag(pcov))
        
        self.fixed_param_indices = fixed_param_values.keys() if fixed_param_values else []
        self.fixed_param_values = fixed_param_values.copy()
        
        # Calculate fitted curve and residuals
        y_fit = model_func(x, *params)
        residuals = y_no_bg - y_fit
        
        # Extract individual peak components
        peak_components = self._extract_peak_components(x, params)
        
        
        
        self.peak_components = peak_components
        
        # Calculate goodness of fit metrics
        n = len(y_no_bg)
        p = len(params)
        
        sse = np.sum(residuals**2)
        sst = np.sum((y_no_bg - np.mean(y_no_bg))**2)
        
        # R-squared
        r_squared = 1 - (sse / sst) if sst > 0 else 0
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1)) if n > p + 1 else 0
        
        # Chi-squared
        denom = np.maximum(y_fit, 1e-10) / (dwell_time * n_scans)
        chi_squared = np.sum(residuals**2 / denom)
        
        # Reduced chi-squared
        red_chi_squared = chi_squared / (n - p) if n > p else np.inf
        
        z_scores = (residuals - np.mean(residuals)) / np.std(residuals) if np.std(residuals) != 0 else np.zeros_like(residuals)
        
        # autocorrelation lag-1
        def lag1_autocorr(a):
            a = a - np.nanmean(a)
            a = a[~np.isnan(a)]
            if len(a) < 2: return np.nan
            return np.corrcoef(a[:-1], a[1:])[0,1]
        
        print("lag-1 autocorr of residuals: ", lag1_autocorr(residuals))
        
        if fixed_param_values:
            # ... after fitting with minimize ...
            
            # Better error estimation using finite differences
            try:
                # Calculate Hessian numerically if not available
                if not hasattr(res, 'hess_inv') or res.hess_inv is None:
                    from scipy.optimize import approx_fprime
                    n_params = len(initial_params)
                    hess = np.zeros((n_params, n_params))
                    eps = 1e-6
                    
                    # Approximate Hessian
                    for i in range(n_params):
                        for j in range(i, n_params):
                            # Finite difference approximation
                            params_i = initial_params.copy()
                            params_j = initial_params.copy()
                            params_i[i] += eps
                            params_j[j] += eps
                            
                            # Calculate gradient differences
                            # (This is simplified - you might want a more robust method)
                            pass
                    
                    cov = np.linalg.inv(hess) * (sse / (n - p))
                else:
                    # Use the inverse Hessian from minimize
                    hess_inv = res.hess_inv.todense()
                    cov = hess_inv * (sse / (n - p))
                
                perr = np.sqrt(np.diag(cov))
                # Handle potential negative values on diagonal
                perr = np.where(perr >= 0, perr, 0)
                
            except Exception as e:
                logger.warning(f"Could not calculate parameter errors: {e}")
                perr = np.full(len(initial_params), np.nan)
        
        # Store results
        self.fit_result = {
            'x': x,
            'y': y,
            'background': background,
            'y_fit': y_fit,
            'params': params,
            'perr': perr,
            'residuals': residuals,
            'peak_components': peak_components
        }
        
        self.fit_result['goodness_of_fit'] = {
            'sse': sse,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'chi_squared': chi_squared,
            'reduced_chi_squared': red_chi_squared,
            'r_mean': np.mean(residuals),
            'r_std': np.std(residuals),
            'lag1_autocorr': pd.Series(z_scores).autocorr(lag=1)
        }
        
        return self
    
    def plot_results(self, fig=None, ax=None, figsize=(10, 8), show_components=True, 
                    show_residuals=True, show_background=True, dpi=100,
                    show_residual_hist=True, bins=30, show_gof=True):
        """Plot the fitting results with optional residual histogram."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        # Extract data
        x = self.fit_result['x']
        y = self.fit_result['y']
        background = self.fit_result['background']
        y_fit = self.fit_result['y_fit']
        residuals = self.fit_result['residuals']
        peak_components = self.fit_result['peak_components']
        
        # Create figure with GridSpec
        if fig is None or ax is None:
            if show_residuals:
                if show_residual_hist:
                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1],
                                        wspace=0.05, hspace=0.05)
                    ax_main = fig.add_subplot(gs[0, 0])
                    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
                    ax_hist = fig.add_subplot(gs[1, 1], sharey=ax_res)
                else:
                    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=figsize, 
                                                        gridspec_kw={'height_ratios': [3, 1]},
                                                        sharex=True, dpi=dpi)
                    ax_hist = None
            else:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax_res, ax_hist = None, None
                ax_main = ax
        else:
            ax_main = ax
            ax_res, ax_hist = None, None
        
        # Plot original data
        ax_main.scatter(x, y, s=20, alpha=0.7, label='Data', color='black')
        ax_main.set_ylim(0.95*np.min(y), 1.05*np.max(y))
        
        # Background
        if show_background and np.any(background != 0):
            ax_main.plot(x, background, '--', color='black', alpha=0.7, lw=3,
                        label=f'Bg: {self.fit_config.background_type.capitalize()}')
        
        # Components
        if show_components:
            for i, component in enumerate(peak_components):
                ax_main.plot(x, component['y_values'] + background, '-', alpha=0.6,
                            label = f"{component.get('name', component['type'])}" \
                            ""
                            # f"{component['params']['center']:.2f} eV"
                            )
                # Vertical line
                # ax_main.axline(xy1=(component['params']['center'], ax_main.get_ylim()[0]),
                #                xy2=(component['params']['center'], component['params']['amplitude']),
                #                linestyle=':', alpha=0.5)
                ax_main.fill_between(x, background, component['y_values'] + background,
                                    alpha=0.25)
        
        # Total fit
        ax_main.plot(x, y_fit + background, 'r-', linewidth=2, label='Fit')
        
        # Residuals
        if show_residuals and ax_res is not None:
            plt.setp(ax_main.get_xticklabels(), visible=False)
            ax_res.plot(x, residuals, 'o-', markersize=3, color='blue')
            ax_res.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # # Add Savitzky-Golay filtered envelope
            # try:
            #     # Determine window length (must be odd and less than data length)
            #     # For X% of data points
            #     window_length = int(len(residuals) * 0.25)

            #     # Make sure it's odd for Savitzky-Golay
            #     if window_length % 2 == 0:
            #         window_length += 1  # Make it odd

            #     # Ensure it's within valid range (at least 5, at most len(residuals))
            #     window_length = max(5, min(window_length, len(residuals)))
                
            #     # Apply Savitzky-Golay filter
            #     smoothed_residuals = savgol_filter(residuals, window_length, 3)  # 3rd order polynomial
                
            #     # Plot the smoothed envelope
            #     ax_res.plot(x, smoothed_residuals, '--', color='red', linewidth=1.5, alpha=0.8, label='Smoothed residuals')
                
            #     # Optionally, you can also plot a shaded region around the smoothed line
            #     residual_std = np.std(residuals - smoothed_residuals)
            #     # ax_res.fill_between(x, smoothed_residuals - 2*residual_std, 
            #     #                    smoothed_residuals + 2*residual_std, 
            #     #                    color='red', alpha=0.1, label='±2σ envelope')
                
            # except Exception as e:
            #     logger.warning(f"Could not apply Savitzky-Golay filter: {e}")
            #     # Fallback: plot simple moving average
            #     window_size = min(10, len(residuals))
            #     if window_size > 0:
            #         weights = np.ones(window_size) / window_size
            #         smoothed_residuals = np.convolve(residuals, weights, mode='same')
            #         ax_res.plot(x, smoothed_residuals, '--', color='grey', linewidth=1.5, alpha=0.8, label='Smoothed residuals')
            
            residual_span = abs(max(residuals) - min(residuals))
            raw_step = residual_span / 5

            # Round to nearest multiple of 10
            step = int(round(raw_step / 10.0) * 10)

            # Fallback: if residuals are very small, avoid step=0
            if step == 0:
                step = max(1, int(round(raw_step)))
            ax_res.set_ylabel('Residuals, counts/s')
            ax_res.set_xlabel('Binding Energy, eV')
            # ax_res.grid(True, alpha=0.3)
            ax_res.yaxis.set_major_locator(
                ticker.MultipleLocator(step)
                )
            ax_res.yaxis.set_minor_locator(
                ticker.MultipleLocator(step / 2)
                )
        
        # Residual histogram
        if show_residual_hist and ax_hist is not None:
            # Histogram
            counts, bin_edges, _ = ax_hist.hist(
                residuals, bins=bins, orientation='horizontal',
                color='blue', alpha=0.7, edgecolor='black', density=True
            )

            # Fit Gaussian
            mu, sigma = np.mean(residuals), np.std(residuals)

            # Compute Gaussian curve
            y_vals = np.linspace(min(residuals), max(residuals), 300)
            pdf_vals = norm.pdf(y_vals, mu, sigma)

            # Normalize to match histogram scaling (density=True handles it)
            ax_hist.plot(pdf_vals, y_vals, 'r-', lw=2, label=f'Gaussian')
            ax_hist.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            ax_hist.set_xlabel("Density, a.u.")
            # ax_hist.grid(True, alpha=0.3)
            ax_hist.legend(loc="lower right", frameon=True, fontsize='x-small')

            # Hide duplicate y ticks
            plt.setp(ax_hist.get_yticklabels(), visible=False)
        
        # Labels and legend
        ax_main.set_ylabel('Intensity, counts/s')
        if not show_residuals:
            ax_main.set_xlabel('Binding Energy, eV')
        # ax_main.grid(which='major', alpha=0.3)
        ax_main.legend(loc='center left', frameon=True, fontsize='x-small')
        ax_main.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        y_span = abs(max(y) - min(y))
        raw_y_step = y_span / 5

        # Round to nearest multiple of 500
        y_step = int(round(raw_y_step / 500.0) * 500)

        # Fallback: if residuals are very small, avoid step=0
        if y_step == 0:
            y_step = max(1, int(round(raw_step)))
        ax_main.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax_main.yaxis.set_minor_locator(ticker.MultipleLocator(y_step/5))
        
        
        
        # Invert X (XPS convention)
        # if x[0] > x[-1]:
        ax_main.invert_xaxis()
        # if ax_res is not None:
        #     ax_res.invert_xaxis()
        
        # Add goodness of fit text
        if 'goodness_of_fit' in self.fit_result and show_gof:
            gof = self.fit_result['goodness_of_fit']
            fit_text = (f"$R^2$ = {gof['r_squared']:.4f}\n"
                        f"Adj. $R^2$ = {gof['adj_r_squared']:.4f}\n"
                        f"Red. $\chi^2$ = {gof['reduced_chi_squared']:.4f}\n"
                        f"R mean = {gof['r_mean']:.3f}, R std = {gof['r_std']:.3f}\n"
                        f"lag-1 autocorr = {gof['lag1_autocorr']:.3f}")
            ax_main.annotate(fit_text, xy=(0.02, 0.97), xycoords='axes fraction',
                            va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7), fontsize='small')
        
        self.last_plot_fig = fig
        # plt.tight_layout()
        return fig, (ax_main, ax_res, ax_hist)
    
    def get_fit_report(self):
        """Generate a detailed fit report with parameter uncertainties."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        report = ["XPS Fitting Report", "=" * 20 + "\n"]
        
        # Add fitting range
        if self.x_fit is not None:
            energy_range = max(self.x_fit) - min(self.x_fit)
            n_points = len(self.x_fit)
            avg_step = np.mean(np.diff(self.x_fit))
            std_step = np.std(np.diff(self.x_fit))
            report.append(f"Fitting E range: {min(self.x_fit):.2f} - {max(self.x_fit):.2f} eV\n")
            report.append(f"Number of E points: {n_points}\n")
            report.append(f"Average E step size (mean+std): {avg_step:.4f} ± {std_step:.4f} eV\n")
            report.append(f"Estimated E resolution (avg step): {avg_step:.4f} eV\n")
            report.append(f"Total E range: {energy_range:.2f} eV\n")
        else:
            report.append("Fitting range: Not available (x_fit is None)\n")
        
        # Add background info
        report.append(f"Background: {self.fit_config.background_type}")
        if self.fit_config.background_params:
            report.append(f"Background parameters: {self.fit_config.background_params}")
        report.append("")
        
        # Get uncertainties (perr)
        perr = self.fit_result.get("perr", None)
        
        # Add peak info
        report.append("Fitted Peaks:")
        report.append("-" * 15)
        # Keep track of parameter indices
        total_param_index = 0  # Index in the FULL parameter list
        free_param_index = 0   # Index in the perr array (only free parameters)
        
        for i, component in enumerate(self.peak_components):
            params = component["params"]
            report.append(f"Peak {i+1} ({component.get('name', 'unknown name')}, {component['type']}):")
            
            for name, value in params.items():
                # Check if this parameter was fixed
                is_fixed = False
                if hasattr(self, 'fixed_param_indices'):
                    is_fixed = total_param_index in self.fixed_param_indices
                
                if not is_fixed and perr is not None and free_param_index < len(perr):
                    err = perr[free_param_index]
                    if err is not None and not np.isnan(err) and err > 0:
                        report.append(f"  {name}: {value:.4f} +/- {err:.4f}")
                    else:
                        report.append(f"  {name}: {value:.4f} (error calculation failed)")
                    free_param_index += 1
                else:
                    if is_fixed:
                        report.append(f"  {name}: {value:.4f} (fixed)")
                    else:
                        report.append(f"  {name}: {value:.4f} (no error)")
                
                total_param_index += 1
            
            # Calculate peak area
            x = self.fit_result["x"]
            y = component["y_values"]
            dx = np.mean(np.diff(x))
            area = np.sum(y) * abs(dx)
            
            report.append(f"  area: {area:.4f}")
            report.append("")
        
        # Add goodness of fit metrics
        if "goodness_of_fit" in self.fit_result:
            report.append("Goodness of Fit:")
            report.append("-" * 15)
            
            gof = self.fit_result["goodness_of_fit"]
            report.append(f"R-squared: {gof['r_squared']:.6f}")
            report.append(f"Adjusted R-squared: {gof['adj_r_squared']:.6f}")
            report.append(f"Chi-squared: {gof['chi_squared']:.6f}")
            report.append(f"Reduced chi-squared: {gof['reduced_chi_squared']:.6f}")
            
            report.append(f"R mean: {gof['r_mean']:.6f}")
            report.append(f"R std: {gof['r_std']:.6f}")
            report.append(f"lag-1 autocorr of residuals: {gof['lag1_autocorr']:.6f}")
        
        return "\n".join(report)

    
    def save_results(self, filename_prefix):
        """Save fitting results to files."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        # Ensure the directory exists
        directory = os.path.dirname(filename_prefix)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save plot
        # Check if we have a stored plot, otherwise create one
        if hasattr(self, 'last_plot_fig') and self.last_plot_fig is not None:
            fig = self.last_plot_fig
            # Optionally, you might want to create a fresh copy to avoid modifying the original
            # fig = self.last_plot_fig  # Just use it as is
        else:
            # Create a new plot
            fig, _ = self.plot_results()
        fig.savefig(f"{filename_prefix}_fit.png", dpi=96, bbox_inches='tight')
        fig.savefig(f"{filename_prefix}_fit.svg", dpi=96, bbox_inches='tight', transparent=True)
        fig.savefig(f"{filename_prefix}_fit.eps", dpi=96, bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        # Save fit report
        report = self.get_fit_report()
        with open(f"{filename_prefix}_report.txt", 'w') as f:
            f.write(report)
        
        # Save data as CSV
        x = self.fit_result['x']
        y = self.fit_result['y']
        background = self.fit_result['background']
        y_fit = self.fit_result['y_fit'] + background
        
        data = {
            'binding_energy': x,
            'intensity': y,
            'background': background,
            'fit': y_fit
        }
        
        # Add individual components
        for i, component in enumerate(self.peak_components):
            data[f'peak_{i+1}'] = component['y_values'] + background
        
        # Convert to CSV
        lines = [','.join(data.keys())]
        for i in range(len(x)):
            line = ','.join([str(data[key][i]) for key in data.keys()])
            lines.append(line)
        
        with open(f"{filename_prefix}_data.csv", 'w') as f:
            f.write('\n'.join(lines))
            
        logger.info(f"Results saved with prefix: {filename_prefix}")
        return self
    
    def calculate_peak_widths(self):
        """Calculate widths for all fitted peaks."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        width_results = {}
        x = self.fit_result['x']
        
        for i, component in enumerate(self.peak_components):
            peak_type = component['type']
            params = component['params']
            
            if peak_type == 'voigt':
                fwhm = calculate_voigt_fwhm(
                    params['amplitude'], params['center'], 
                    params['fwhm_g'], params['fwhm_l'], x_range=x
                )
                width_results[f'peak_{i+1}'] = {'type': 'voigt', 'fwhm': fwhm}
                
            elif peak_type == 'doniach_sunjic':
                left_hwhm, right_hwhm = calculate_doniach_sunjic_widths(
                    params['amplitude'], params['center'], 
                    params['fwhm'], params['asymmetry'], x_range=x
                )
                width_results[f'peak_{i+1}'] = {
                    'type': 'doniach_sunjic', 
                    'left_hwhm': left_hwhm, 
                    'right_hwhm': right_hwhm
                }
                
            elif peak_type == 'asymmetric_voigt':
                fwhm, left_hwhm, right_hwhm = calculate_asymmetric_voigt_fwhm(
                    params['amplitude'], params['center'], 
                    params['fwhm_g'], params['fwhm_l'], params['asymmetry'], x_range=x
                )
                width_results[f'peak_{i+1}'] = {
                    'type': 'asymmetric_voigt', 
                    'fwhm': fwhm,
                    'left_hwhm': left_hwhm, 
                    'right_hwhm': right_hwhm
                }
        
        return width_results

class XPSSpectrum:
    def __init__(self, binding_energy, counts_per_second):
        self.binding_energy = binding_energy
        self.counts_per_second = counts_per_second
        self.filtered_counts = None  # Store filtered data
        self.filter_type = None      # Store filter type used
    
    def fft_filter(self, filter_type='lowpass', cutoff_freq=None, cutoff_fraction=0.1, 
                window_type='hann', pad_factor=2, padding_mode='symmetric'):
        """
        Apply FFT-based filtering to the spectrum.
        
        Parameters:
        -----------
        filter_type : str
            'lowpass', 'highpass', 'bandpass', or 'bandstop'
        cutoff_freq : float or tuple
            For lowpass/highpass: single frequency (in 1/eV units)
            For bandpass/bandstop: tuple of (low_cutoff, high_cutoff)
        cutoff_fraction : float
            Fraction of Nyquist frequency to use as cutoff if cutoff_freq not specified
        window_type : str
            Window function for smoothing cutoff: 'rect', 'hann', 'hamming', 'blackman'
        pad_factor : int
            Zero-padding factor for FFT (helps with edge effects)
        padding_mode : str
            'symmetric', 'reflect', 'edge', 'constant' - how to pad data
        """
        y = self.counts_per_second.copy()
        x = self.binding_energy
        
        # Sort data if not in descending order (XPS convention)
        if x[0] < x[-1]:
            sort_idx = np.argsort(x)[::-1]
            x = x[sort_idx]
            y = y[sort_idx]
        
        # Calculate sampling interval (in eV)
        dx = np.mean(np.diff(x))
        
        # Calculate Nyquist frequency (in 1/eV)
        nyquist_freq = 1 / (2 * dx)
        
        # Determine cutoff frequencies
        if cutoff_freq is None:
            if filter_type in ['lowpass', 'highpass']:
                cutoff_freq = cutoff_fraction * nyquist_freq
            else:
                # Default band: remove middle frequencies
                cutoff_freq = (0.1 * nyquist_freq, 0.4 * nyquist_freq)
        
        print(f"Filter parameters:")
        print(f"  Sampling interval (dx): {dx:.6f} eV")
        print(f"  Nyquist frequency: {nyquist_freq:.4f} 1/eV")
        print(f"  Cutoff frequency: {cutoff_freq} 1/eV")
        print(f"  Cutoff fraction: {cutoff_fraction}")
        
        # Apply symmetric padding to reduce edge effects
        n_original = len(y)
        n_padded = n_original * pad_factor
        pad_before = (n_padded - n_original) // 2
        pad_after = n_padded - n_original - pad_before
        
        # Use symmetric padding (best for FFT)
        if padding_mode == 'symmetric':
            y_padded = np.pad(y, (pad_before, pad_after), mode='symmetric')
        elif padding_mode == 'reflect':
            y_padded = np.pad(y, (pad_before, pad_after), mode='reflect')
        elif padding_mode == 'edge':
            y_padded = np.pad(y, (pad_before, pad_after), mode='edge')
        else:
            y_padded = np.pad(y, (pad_before, pad_after), mode='constant', constant_values=0)
        
        # Perform FFT
        y_fft = np.fft.rfft(y_padded)
        freqs = np.fft.rfftfreq(n_padded, d=dx)
        
        # Create filter mask
        mask = self._create_fft_filter_mask(freqs, filter_type, cutoff_freq, window_type)
        
        # Apply filter
        y_fft_filtered = y_fft * mask
        
        # Inverse FFT
        y_filtered_padded = np.fft.irfft(y_fft_filtered, n=n_padded)
        
        # Remove padding - take the middle portion
        y_filtered = y_filtered_padded[pad_before:pad_before + n_original]
        
        # Store results
        self.filtered_counts = y_filtered
        self.filter_type = filter_type
        self.filter_params = {
            'cutoff_freq': cutoff_freq,
            'window_type': window_type,
            'nyquist_freq': nyquist_freq,
            'cutoff_fraction': cutoff_fraction
        }
        
        # For debugging
        self.debug_info = {
            'y_padded': y_padded,
            'y_filtered_padded': y_filtered_padded,
            'freqs': freqs,
            'mask': mask,
            'y_fft_original': y_fft,
            'y_fft_filtered': y_fft_filtered
        }
        
        return y_filtered
    
    def _create_fft_filter_mask(self, freqs, filter_type, cutoff_freq, window_type):
        """Create the frequency domain filter mask."""
        mask = np.ones_like(freqs, dtype=float)
        
        # Window function for smooth transition
        if window_type == 'hann':
            window_func = lambda x: 0.5 * (1 - np.cos(np.pi * x))
        elif window_type == 'hamming':
            window_func = lambda x: 0.54 - 0.46 * np.cos(np.pi * x)
        elif window_type == 'blackman':
            window_func = lambda x: (0.42 - 0.5 * np.cos(np.pi * x) + 
                                    0.08 * np.cos(2 * np.pi * x))
        else:  # rectangular (sharp cutoff)
            window_func = lambda x: 1.0
        
        # Apply filter based on type
        if filter_type == 'lowpass':
            # Lowpass: keep frequencies below cutoff
            if isinstance(cutoff_freq, (tuple, list)):
                cutoff = cutoff_freq[0]
            else:
                cutoff = cutoff_freq
            
            # Create transition region
            transition_width = cutoff * 0.1
            low_transition = cutoff - transition_width/2
            high_transition = cutoff + transition_width/2
            
            for i, f in enumerate(freqs):
                if f <= low_transition:
                    mask[i] = 1.0
                elif f <= high_transition:
                    # Apply window function in transition region
                    x_norm = (f - low_transition) / (transition_width)
                    mask[i] = window_func(1 - x_norm)
                else:
                    mask[i] = 0.0
        
        elif filter_type == 'highpass':
            # Highpass: keep frequencies above cutoff
            if isinstance(cutoff_freq, (tuple, list)):
                cutoff = cutoff_freq[0]
            else:
                cutoff = cutoff_freq
            
            transition_width = cutoff * 0.1
            low_transition = cutoff - transition_width/2
            high_transition = cutoff + transition_width/2
            
            for i, f in enumerate(freqs):
                if f <= low_transition:
                    mask[i] = 0.0
                elif f <= high_transition:
                    x_norm = (f - low_transition) / (transition_width)
                    mask[i] = window_func(x_norm)
                else:
                    mask[i] = 1.0
        
        elif filter_type == 'bandpass':
            # Bandpass: keep frequencies between cutoffs
            low_cut, high_cut = cutoff_freq
            
            # Lower transition
            low_transition_width = low_cut * 0.1
            low_trans_start = low_cut - low_transition_width/2
            low_trans_end = low_cut + low_transition_width/2
            
            # Upper transition
            high_transition_width = high_cut * 0.1
            high_trans_start = high_cut - high_transition_width/2
            high_trans_end = high_cut + high_transition_width/2
            
            for i, f in enumerate(freqs):
                if f <= low_trans_start:
                    mask[i] = 0.0
                elif f <= low_trans_end:
                    x_norm = (f - low_trans_start) / low_transition_width
                    mask[i] = window_func(x_norm)
                elif f <= high_trans_start:
                    mask[i] = 1.0
                elif f <= high_trans_end:
                    x_norm = 1 - (f - high_trans_start) / high_transition_width
                    mask[i] = window_func(x_norm)
                else:
                    mask[i] = 0.0
        
        elif filter_type == 'bandstop':
            # Bandstop: remove frequencies between cutoffs
            low_cut, high_cut = cutoff_freq
            
            # Similar to bandpass but inverted
            low_transition_width = low_cut * 0.1
            low_trans_start = low_cut - low_transition_width/2
            low_trans_end = low_cut + low_transition_width/2
            
            high_transition_width = high_cut * 0.1
            high_trans_start = high_cut - high_transition_width/2
            high_trans_end = high_cut + high_transition_width/2
            
            for i, f in enumerate(freqs):
                if f <= low_trans_start:
                    mask[i] = 1.0
                elif f <= low_trans_end:
                    x_norm = 1 - (f - low_trans_start) / low_transition_width
                    mask[i] = window_func(x_norm)
                elif f <= high_trans_start:
                    mask[i] = 0.0
                elif f <= high_trans_end:
                    x_norm = (f - high_trans_start) / high_transition_width
                    mask[i] = window_func(x_norm)
                else:
                    mask[i] = 1.0
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return mask
    
    def plot_fft_analysis(self, figsize=(12, 8)):
        """Plot the FFT analysis results."""
        if self.filtered_counts is None:
            raise ValueError("No filtered data available. Run fft_filter() first.")
        
        x = self.binding_energy
        y_original = self.counts_per_second
        y_filtered = self.filtered_counts
        
        # Sort if needed
        if x[0] < x[-1]:
            sort_idx = np.argsort(x)[::-1]
            x = x[sort_idx]
            y_original = y_original[sort_idx]
            y_filtered = y_filtered[sort_idx]
        
        dx = np.mean(np.diff(x))
        
        # Calculate FFTs
        n = len(y_original)
        y_fft_original = np.fft.rfft(y_original)
        y_fft_filtered = np.fft.rfft(y_filtered)
        freqs = np.fft.rfftfreq(n, d=dx)
        
        # Power spectra
        power_original = np.abs(y_fft_original)**2
        power_filtered = np.abs(y_fft_filtered)**2
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Original vs Filtered Spectrum
        ax1 = axes[0, 0]
        ax1.plot(x, y_original, 'b-', alpha=0.7, label='Original')
        ax1.plot(x, y_filtered, 'r-', alpha=0.9, label='Filtered')
        ax1.set_xlabel('Binding Energy (eV)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'{self.filter_type.capitalize()} Filtered Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if x[0] < x[-1]:
            ax1.invert_xaxis()
        
        # Plot 2: Difference
        ax2 = axes[0, 1]
        difference = y_original - y_filtered
        ax2.plot(x, difference, 'g-', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Binding Energy (eV)')
        ax2.set_ylabel('Difference (Original - Filtered)')
        ax2.set_title('Removed Noise Component')
        ax2.grid(True, alpha=0.3)
        if x[0] < x[-1]:
            ax2.invert_xaxis()
        
        # Plot 3: Power Spectrum (log scale)
        ax3 = axes[0, 2]
        ax3.plot(freqs[1:], power_original[1:], 'b-', alpha=0.7, label='Original')
        ax3.plot(freqs[1:], power_filtered[1:], 'r-', alpha=0.7, label='Filtered')
        if hasattr(self, 'filter_params'):
            cutoff = self.filter_params['cutoff_freq']
            if isinstance(cutoff, tuple):
                ax3.axvline(cutoff[0], color='k', linestyle='--', alpha=0.5)
                ax3.axvline(cutoff[1], color='k', linestyle='--', alpha=0.5)
            else:
                ax3.axvline(cutoff, color='k', linestyle='--', alpha=0.5, label='Cutoff')
        ax3.set_xlabel('Frequency (1/eV)')
        ax3.set_ylabel('Power (log scale)')
        ax3.set_yscale('log')
        ax3.set_title('Power Spectrum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Zoomed power spectrum
        ax4 = axes[1, 0]
        max_freq_to_show = min(1.0, freqs[-1])  # Show up to 1 eV^-1 or max available
        idx = freqs <= max_freq_to_show
        ax4.plot(freqs[idx][1:], power_original[idx][1:], 'b-', alpha=0.7, label='Original')
        ax4.plot(freqs[idx][1:], power_filtered[idx][1:], 'r-', alpha=0.7, label='Filtered')
        if hasattr(self, 'filter_params'):
            cutoff = self.filter_params['cutoff_freq']
            if isinstance(cutoff, tuple):
                ax4.axvline(cutoff[0], color='k', linestyle='--', alpha=0.5)
                ax4.axvline(cutoff[1], color='k', linestyle='--', alpha=0.5)
            else:
                ax4.axvline(cutoff, color='k', linestyle='--', alpha=0.5, label='Cutoff')
        ax4.set_xlabel('Frequency (1/eV)')
        ax4.set_ylabel('Power')
        ax4.set_title('Power Spectrum (Zoomed)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Phase spectrum
        ax5 = axes[1, 1]
        phase_original = np.angle(y_fft_original)
        phase_filtered = np.angle(y_fft_filtered)
        ax5.plot(freqs[1:], phase_original[1:], 'b-', alpha=0.7, label='Original')
        ax5.plot(freqs[1:], phase_filtered[1:], 'r-', alpha=0.7, label='Filtered')
        ax5.set_xlabel('Frequency (1/eV)')
        ax5.set_ylabel('Phase (rad)')
        ax5.set_title('Phase Spectrum')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Residuals histogram
        ax6 = axes[1, 2]
        ax6.hist(difference, bins=50, alpha=0.7, edgecolor='black')
        ax6.axvline(x=0, color='r', linestyle='--')
        ax6.set_xlabel('Residual Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'Residuals Distribution\nMean: {np.mean(difference):.3f}, Std: {np.std(difference):.3f}')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    

if __name__ == "__main__":
    # Sample XPS data
    
    # Example usage
    is_example = False
    if is_example:
        # Generate synthetic data for a C 1s spectrum
        be = np.linspace(290, 280, 500)  # Binding energy in eV
        
        # Create synthetic peaks
        peak1 = gaussian(be, 1000, 284.8, 1.1)  # sp2 carbon
        peak2 = gaussian(be, 300, 286.3, 1.3)   # C-O
        peak3 = gaussian(be, 200, 288.2, 1.5)   # C=O
        
        # Add noise and background
        noise = np.random.normal(0, 30, size=len(be))
        background = 200 - (be - 280) * 10
        counts = peak1 + peak2 + peak3 + background + noise
        
        # Create spectrum object
        spectrum = XPSSpectrum(be, counts)
        
        # Create and set up fitter
        fitter = XPSFitter(spectrum)
        
        # Create configuration dictionary
        config_dict = {
            'energy_range': [282, 289],
            'background_type': 'shirley',
            'background_params': {'tolerance': 1e-6, 'max_iterations': 100},
            'peaks': [
                {
                    'type': 'gaussian',
                    'initial_amplitude': 1000,
                    'initial_center': 284.8,
                    'initial_fwhm': 1.0,
                    'center_bounds': [284.4, 285.2],
                    'fwhm_bounds': [0.5, 2.0]
                },
                {
                    'type': 'gaussian',
                    'initial_amplitude': 300,
                    'initial_center': 286.3,
                    'initial_fwhm': 1.2,
                    'center_bounds': [285.8, 286.8],
                    'fwhm_bounds': [0.5, 2.0]
                },
                {
                    'type': 'gaussian',
                    'initial_amplitude': 200,
                    'initial_center': 288.1,
                    'initial_fwhm': 1.4,
                    'center_bounds': [287.5, 288.5],
                    'fwhm_bounds': [0.5, 2.5]
                }
            ],
            'max_iterations': 2000,
            'ftol': 1e-10,
            'method': 'trf' # lm only works when you do not specify bounds for the fit parameters
        }
        
        # Load configuration and fit
        fitter.load_config_from_dict(config_dict).fit()
        
        # Display results
        print(fitter.get_fit_report())
        
        # Plot and save results
        fig, axes = fitter.plot_results(figsize=(10, 8), show_components=True)
        # plt.savefig('tests/Specs-xy-data/fit_example/c1s_fit_example2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save all results to files
        # fitter.save_results('tests/Specs-xy-data/fit_example/c1s_fit_example2')
        
        
        # with open('c1s_fit_config.yaml', 'w') as f:
        #     yaml.dump(config_dict, f)
        # Example of how to load configuration from a file
        """
        # Save the configuration to a YAML file for future use
        with open('c1s_fit_config.yaml', 'w') as f:
            yaml.dump(config_dict, f)
        
        # Later, load the configuration from the file
        fitter = XPSFitter(spectrum)
        fitter.load_config_from_file('c1s_fit_config.yaml').fit()
        """
    
    # Insert here your own .csv data file and configuration file
    
    # Si2p
    # config_path = 'configs/Si2p_fit_config_no_constr.yaml'
    # file_path = 'tests/20241011_2/output_data/Si2p.xy_Spectrum_6.csv'
    
    # C1s
    # config_path = 'configs/C1s_fit_config_no_constr.yaml'
    # file_path = 'tests/20241011_2/output_data/C1s.xy_Spectrum_5.csv'
    
    # C1s_reference
    config_path = 'tests/configs/C1s_fit_config_4.yaml'
    file_path = 'tests/20241011_2/output_data/C1s_reference.xy_C1s_2.csv'
    
    csvFile, metadata = read_xps_csv(file_path)
    
    be = csvFile['Binding Energy'].to_numpy()
    counts = csvFile['Counts per Second'].to_numpy()
    
    
    # Do not normalize your data before fitting, unless you also rescale the variances consistently. 
    # Reduced chi-squared is only interpretable when your denominators reflect the true noise model in the same units as the data.
    
    # counts -= np.min(counts)
    # counts /= np.max(counts)
    
    # counts = savgol_filter(counts, 7, 2)  # Smooth data with Savitzky-Golay filter to reduce noise
    
    # Create spectrum with original data
    spectrum = XPSSpectrum(be, counts)

    is_fft = False
    if is_fft:
        # Apply FFT filter (modifies spectrum internally)
        filtered_counts = spectrum.fft_filter(filter_type='lowpass', cutoff_fraction=0.4)
        # Now spectrum has:
        # - spectrum.counts_per_second: original data
        # - spectrum.filtered_counts: filtered data
        # - spectrum.filter_type: 'lowpass'

        # Visualize FFT analysis
        fig = spectrum.plot_fft_analysis()
        plt.show()

        # To fit with filtered data, you need to create a new XPSSpectrum object
        # OR modify XPSFitter to use filtered_counts
        spectrum_for_fitting = XPSSpectrum(be, counts - filtered_counts)  # Create new with filtered data
        fitter = XPSFitter(spectrum_for_fitting)
    else: fitter = XPSFitter(spectrum)
    
    fitter.load_config_from_file(config_path).fit(dwell_time=metadata['dwell_time'], n_scans=metadata['n_scans'])
    
    # Print results in terminal
    fit_report = fitter.get_fit_report()
    print(fit_report)
    
    # # Area ratio for buffer layer analysis in C1s range
    # # Find all area values in the report
    # area_matches = re.findall(r'area: (\d+\.\d+)', fit_report)
    # if len(area_matches) >= 2:
    #     # Convert to floats
    #     areas = [float(area) for area in area_matches]
        
    #     # Get the last two areas
    #     last_area = areas[-1]
    #     second_last_area = areas[-2]
        
    #     # Calculate ratio
    #     if second_last_area > 0:
    #         ratio = last_area / (second_last_area + last_area) *100
    #         print(f"\nPeak Area Ratio : {ratio:.2f}% ({100-ratio:.2f}%)")
    #     else:
    #         print("\nCannot calculate ratio: division by zero")
    # else:
    #     print("\nNot enough peaks found to calculate ratio")
    
    # print('\n', fitter.calculate_peak_widths())
    
    
    # Show results in a plot
    fig, axes = fitter.plot_results(figsize=(10, 8), show_components=True,
                                    show_residuals=False, show_residual_hist=True,
                                    show_gof=False
                                    )
    # plt.savefig('tests/result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # # Save all results to files
    fitter.save_results('tests/C_1s_reference_fit/C_1s_4fit_260213')