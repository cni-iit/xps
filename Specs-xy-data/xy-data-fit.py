import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy import integrate
import yaml
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('XPSFit')

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
    """Voigt peak function (convolution of Gaussian and Lorentzian)."""
    # This is an approximation of the Voigt profile
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2
    
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(np.exp(-z**2) * (1 + np.math.erf(-1j * z)))

def doniach_sunjic(x, amplitude, center, fwhm, asymmetry):
    """Doniach-Sunjic asymmetric line shape for XPS."""
    gamma = fwhm / 2
    arg = (x - center) / gamma
    
    # Handle potential numerical issues
    safe_arg = np.where(np.abs(arg) < 1e-10, 1e-10, arg)
    
    # Calculate the DS function with asymmetry parameter
    ds = np.cos(np.pi * asymmetry / 2 + (1 - asymmetry) * np.arctan(safe_arg))
    ds *= np.power(1 + arg**2, (1 - asymmetry) / 2)
    
    return amplitude * ds / (1 + arg**2)**(0.5)

def asymmetric_voigt(x, amplitude, center, fwhm_g, fwhm_l, asymmetry):
    """Asymmetric Voigt function by combining Voigt with Doniach-Sunjic."""
    v = voigt(x, 1.0, center, fwhm_g, fwhm_l)
    ds = doniach_sunjic(x, 1.0, center, (fwhm_g + fwhm_l)/2, asymmetry)
    
    # Normalize
    v_max = np.max(v)
    ds_max = np.max(ds)
    
    if v_max > 0 and ds_max > 0:
        v = v / v_max
        ds = ds / ds_max
        
    # Mix based on asymmetry parameter
    mix = v * (1 - abs(asymmetry)/2) + ds * (abs(asymmetry)/2)
    return amplitude * mix / np.max(mix)

# Background functions
def linear_background(x, slope, intercept):
    """Linear background."""
    return slope * x + intercept

def shirley_background(x, y, tol=1e-5, max_iter=50):
    """
    Calculate iterative Shirley background.
    
    Args:
        x: binding energy array (should be in descending order for XPS)
        y: intensity array
        tol: convergence tolerance
        max_iter: maximum number of iterations
        
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
    
    # Initial background
    background = np.ones_like(y) * y[-1]
    
    # Iterative procedure
    for _ in range(max_iter):
        # Calculate the integral of spectrum above background
        integral = np.zeros_like(y)
        for i in range(len(y)-1, -1, -1):  # Backward iteration
            if i < len(y) - 1:
                integral[i] = integral[i+1] + (y[i] - background[i] + y[i+1] - background[i+1]) * (x[i] - x[i+1]) / 2
        
        # Normalize the integral
        integral = integral / integral[0] if integral[0] > 0 else integral
        
        # Calculate new background
        new_background = y[-1] + (y[0] - y[-1]) * integral
        
        # Check convergence
        if np.max(np.abs(new_background - background)) < tol:
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

@dataclass
class PeakConfig:
    """Configuration for a peak in XPS fitting."""
    peak_type: str  # 'gaussian', 'lorentzian', 'voigt', 'doniach_sunjic', 'asymmetric_voigt'
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
    
    # Background configuration
    background_type: str  # 'linear', 'shirley', 'tougaard', 'none'
    background_params: Dict = None
    
    # Peaks configuration
    peaks: List[PeakConfig]
    
    # Additional fitting options
    max_iterations: int = 1000
    ftol: float = 1e-8
    method: str = 'lm'  # 'lm', 'trf', 'dogbox'

class XPSFitter:
    def __init__(self, spectrum=None):
        """Initialize the XPS fitter with an optional spectrum."""
        self.spectrum = spectrum
        self.fit_config = None
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
            method=config_dict.get('method', 'lm')
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
                
            param_idx += n_params
            
        return components
    
    def fit(self):
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
        
        # If we have fixed parameters, use a different approach
        if fixed_param_values:
            # Create objective function that handles fixed parameters
            obj_func = self._create_objective_function_with_fixed_params(
                model_func, x, y_no_bg, fixed_param_values)
            
            # Use minimize instead of curve_fit for more flexibility
            res = minimize(
                lambda p: np.sum(obj_func(p)**2),
                initial_params,
                method='BFGS',
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
            perr = np.zeros_like(params)  # No standard errors in this case
            
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
        
        # Calculate fitted curve and residuals
        y_fit = model_func(x, *params)
        residuals = y_no_bg - y_fit
        
        # Extract individual peak components
        peak_components = self._extract_peak_components(x, params)
        
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
        chi_squared = np.sum((residuals**2) / np.abs(y_fit))
        red_chi_squared = chi_squared / (n - p) if n > p else np.inf
        
        self.fit_result['goodness_of_fit'] = {
            'sse': sse,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'chi_squared': chi_squared,
            'reduced_chi_squared': red_chi_squared
        }
        
        return self
    
    def plot_results(self, fig=None, ax=None, figsize=(10, 8), show_components=True, 
                     show_residuals=True, show_background=True, dpi=100):
        """Plot the fitting results."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        # Extract data
        x = self.fit_result['x']
        y = self.fit_result['y']
        background = self.fit_result['background']
        y_fit = self.fit_result['y_fit']
        residuals = self.fit_result['residuals']
        peak_components = self.fit_result['peak_components']
        
        # Create figure
        if fig is None or ax is None:
            if show_residuals:
                fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=figsize, 
                                                    gridspec_kw={'height_ratios': [3, 1]},
                                                    sharex=True, dpi=dpi)
                ax = ax_main
            else:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax_res = None
        
        # Plot original data
        ax.scatter(x, y, s=20, alpha=0.7, label='Data', color='black')
        
        # Plot background if requested
        if show_background and np.any(background != 0):
            ax.plot(x, background, '--', color='gray', alpha=0.7, label='Background')
        
        # Plot individual components if requested
        if show_components:
            for i, component in enumerate(peak_components):
                ax.plot(x, component['y_values'] + background, '-', alpha=0.6, 
                      label=f"{component['type']} at {component['params']['center']:.2f} eV")
        
        # Plot total fit
        ax.plot(x, y_fit + background, 'r-', linewidth=2, label='Fit')
        
        # Plot residuals if requested
        if show_residuals and ax_res is not None:
            ax_res.plot(x, residuals, 'o-', markersize=3, color='blue')
            ax_res.axhline(y=0, color='r', linestyle='-', alpha=0.5)
            ax_res.set_ylabel('Residuals')
            ax_res.set_xlabel('Binding Energy (eV)')
            ax_res.grid(True, alpha=0.3)
        
        # Add labels and legend
        ax.set_ylabel('Intensity (a.u.)')
        if not show_residuals:
            ax.set_xlabel('Binding Energy (eV)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True)
        
        # XPS convention: higher binding energy on left
        if x[0] < x[-1]:
            ax.invert_xaxis()
        
        # Add goodness of fit text
        if 'goodness_of_fit' in self.fit_result:
            gof = self.fit_result['goodness_of_fit']
            fit_text = (f"R² = {gof['r_squared']:.4f}\n"
                        f"Adj. R² = {gof['adj_r_squared']:.4f}\n"
                        f"Red. χ² = {gof['reduced_chi_squared']:.4f}")
            ax.annotate(fit_text, xy=(0.02, 0.97), xycoords='axes fraction',
                      va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.tight_layout()
        return fig, ax if not show_residuals else (ax, ax_res)
    
    def get_fit_report(self):
        """Generate a detailed fit report."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        report = ["XPS Fitting Report", "=" * 20 + "\n"]
        
        # Add fitting range
        report.append(f"Fitting range: {min(self.x_fit):.2f} - {max(self.x_fit):.2f} eV\n")
        
        # Add background info
        report.append(f"Background: {self.fit_config.background_type}")
        if self.fit_config.background_params:
            report.append(f"Background parameters: {self.fit_config.background_params}")
        report.append("")
        
        # Add peak info
        report.append("Fitted Peaks:")
        report.append("-" * 15)
        
        for i, component in enumerate(self.peak_components):
            params = component['params']
            report.append(f"Peak {i+1} ({component['type']}):")
            
            for name, value in params.items():
                report.append(f"  {name}: {value:.4f}")
                
            # Calculate peak area
            x = self.fit_result['x']
            y = component['y_values']
            dx = np.mean(np.diff(x))
            area = np.sum(y) * abs(dx)
            
            report.append(f"  area: {area:.4f}")
            report.append("")
        
        # Add goodness of fit metrics
        if 'goodness_of_fit' in self.fit_result:
            report.append("Goodness of Fit:")
            report.append("-" * 15)
            
            gof = self.fit_result['goodness_of_fit']
            report.append(f"R-squared: {gof['r_squared']:.6f}")
            report.append(f"Adjusted R-squared: {gof['adj_r_squared']:.6f}")
            report.append(f"Chi-squared: {gof['chi_squared']:.6f}")
            report.append(f"Reduced chi-squared: {gof['reduced_chi_squared']:.6f}")
        
        return "\n".join(report)
    
    def save_results(self, filename_prefix):
        """Save fitting results to files."""
        if self.fit_result is None:
            raise ValueError("No fit results available. Run fit() first.")
        
        # Save plot
        fig, _ = self.plot_results()
        fig.savefig(f"{filename_prefix}_fit.png", dpi=300, bbox_inches='tight')
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


# Example usage
if __name__ == "__main__":
    # Sample XPS data
    class XPSSpectrum:
        def __init__(self, binding_energy, counts_per_second):
            self.binding_energy = binding_energy
            self.counts_per_second = counts_per_second
    
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
        'method': 'lm'
    }
    
    # Load configuration and fit
    fitter.load_config_from_dict(config_dict).fit()
    
    # Display results
    print(fitter.get_fit_report())
    
    # Plot and save results
    fig, axes = fitter.plot_results(figsize=(10, 8), show_components=True)
    plt.savefig('c1s_fit_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save all results to files
    fitter.save_results('c1s_fit')
    
    # Example of how to load configuration from a file
    """
    # Save the configuration to a YAML file for future use
    import yaml
    with open('c1s_fit_config.yaml', 'w') as f:
        yaml.dump(config_dict, f)
    
    # Later, load the configuration from the file
    fitter = XPSFitter(spectrum)
    fitter.load_config_from_file('c1s_fit_config.yaml').fit()
    """