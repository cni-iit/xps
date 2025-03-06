import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Spectrum:
    metadata: Dict[str, str] = field(default_factory=dict)
    binding_energy: List[float] = field(default_factory=list)
    counts_per_second: List[float] = field(default_factory=list)
    spectrum_id: Optional[str] = None
    region: Optional[str] = None

    def plot(self, ax=None, label=None, **kwargs):
        """Plot the spectrum data."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if label is None:
            if self.region and self.spectrum_id:
                label = f"{self.region} (ID: {self.spectrum_id})"
            elif self.region:
                label = self.region
            elif self.spectrum_id:
                label = f"Spectrum ID: {self.spectrum_id}"
            else:
                label = "Spectrum"
                
        ax.plot(self.binding_energy, self.counts_per_second, label=label, **kwargs)
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Counts per Second')
        ax.invert_xaxis()  # XPS convention: binding energy decreases from left to right
        
        return ax


class XPSData:
    def __init__(self):
        self.general_metadata = {}
        self.spectra = []
        
    def parse_file(self, file_path):
        """Parse an XPS .xy file."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Process the file
        current_section = "general_metadata"
        current_spectrum = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if we're in the metadata section
            if line.startswith('#'):
                # Remove the '# ' prefix if it exists
                cleaned_line = line[2:] if line.startswith('# ') else line[1:]
                
                # Check for section transitions
                if "Region:" in cleaned_line:
                    # We're starting a new spectrum section
                    if current_spectrum is not None:
                        self.spectra.append(current_spectrum)
                    
                    current_spectrum = Spectrum()
                    current_section = "spectrum_metadata"
                
                # Process metadata based on current section
                if current_section == "general_metadata":
                    self._process_metadata_line(cleaned_line, self.general_metadata)
                
                elif current_section == "spectrum_metadata" and current_spectrum is not None:
                    self._process_metadata_line(cleaned_line, current_spectrum.metadata)
                    
                    # Extract important values
                    if "Region:" in cleaned_line:
                        current_spectrum.region = cleaned_line.split(":", 1)[1].strip()
                    elif "Spectrum ID:" in cleaned_line:
                        current_spectrum.spectrum_id = cleaned_line.split(":", 1)[1].strip()
                    
                # Check if we're transitioning to data section
                if "ColumnLabels:" in cleaned_line:
                    current_section = "data"
            
            # Process data points
            elif current_section == "data" and current_spectrum is not None:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        binding_energy = float(parts[0])
                        counts = float(parts[1])
                        current_spectrum.binding_energy.append(binding_energy)
                        current_spectrum.counts_per_second.append(counts)
                except ValueError:
                    # Skip lines that can't be parsed as data points
                    pass
        
        # Add the last spectrum if it exists
        if current_spectrum is not None:
            self.spectra.append(current_spectrum)
            
        return self
    
    def _process_metadata_line(self, line, metadata_dict):
        """Process a metadata line and add it to the appropriate dictionary."""
        if ":" in line:
            key, value = line.split(":", 1)
            metadata_dict[key.strip()] = value.strip()
    
    def plot_all_spectra(self, title=None, figsize=(12, 8)):
        """Plot all spectra in the dataset."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, spectrum in enumerate(self.spectra):
            spectrum.plot(ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('XPS Spectra')
            
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    def get_spectrum_by_id(self, spectrum_id):
        """Get a spectrum by its ID."""
        for spectrum in self.spectra:
            if spectrum.spectrum_id == str(spectrum_id):
                return spectrum
        return None
    
    def get_spectrum_by_region(self, region_name):
        """Get spectra by region name."""
        return [s for s in self.spectra if s.region == region_name]
    
    def save_to_csv(self, output_dir, base_filename=None):
        """Save all spectra to individual CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if base_filename is None:
            base_filename = "spectrum"
            
        for i, spectrum in enumerate(self.spectra):
            # Create a meaningful filename
            if spectrum.region and spectrum.spectrum_id:
                filename = f"{base_filename}_{spectrum.region}_{spectrum.spectrum_id}.csv"
            elif spectrum.region:
                filename = f"{base_filename}_{spectrum.region}_{i}.csv"
            elif spectrum.spectrum_id:
                filename = f"{base_filename}_{spectrum.spectrum_id}.csv"
            else:
                filename = f"{base_filename}_{i}.csv"
                
            # Replace any characters that might not be valid in filenames
            filename = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)
            
            # Write to CSV
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                # Write metadata as comments
                f.write("# XPS Spectrum Data\n")
                for key, value in spectrum.metadata.items():
                    f.write(f"# {key}: {value}\n")
                
                # Write column headers and data
                f.write("Binding Energy,Counts per Second\n")
                for energy, counts in zip(spectrum.binding_energy, spectrum.counts_per_second):
                    f.write(f"{energy},{counts}\n")


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "prova.xy"
    
    xps_data = XPSData()
    xps_data.parse_file(file_path)
    
    print(f"Loaded {len(xps_data.spectra)} spectra")
    print("\nGeneral Metadata:")
    for key, value in xps_data.general_metadata.items():
        print(f"  {key}: {value}")
    
    # Example: Plot all spectra
    fig, ax = xps_data.plot_all_spectra()
    plt.show()
    
    # Example: Get a specific spectrum and plot it
    spec = xps_data.get_spectrum_by_region("Survey")
    if spec:
        spec[0].plot()
        plt.title(f"Survey Spectrum (ID: {spec[0].spectrum_id})")
        plt.show()
    
    # Example: Save to CSV
    xps_data.save_to_csv("output_data")