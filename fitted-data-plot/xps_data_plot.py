import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # Changed from tqdm.notebook for script compatibility

def read_spectral_data(file_path):
    """
    Read spectral data from a csv file and return it as a Pandas DataFrame.
    
    Parameters:
    file_path : str or Path
        Path to the data file
    
    Returns:
    tuple: (fit_summary, data_df)
        fit_summary: Dictionary with region name and acquisition energy,
            and then names, centers, FWHMs, areas and lineshapes of the components of the spectrum
        data_df: DataFrame with counts, components, background, envelope and binding energies
    """
    # Read the first lines to get spectrum metadata
    with open(file_path, 'r') as f:
        region = f.readline().split(':')[2].rstrip().rstrip('"')   # example of 1st line: Cycle 0:Spectrum Group:Ni2p
        acq_energy = np.asarray(f.readline().strip(',').split(',')[1], dtype=float)   # example of 2nd line: ,Characteristic Energy eV,1.486610e+003,Acquisition Time s,1.920000e+000
        names = f.readline().rstrip().rstrip(',\"').replace('Name,,\"', '').split('\",\"')
        centers = np.asarray(f.readline().rstrip().rstrip(',\"').replace('Position,,\"', '').split('\",\"'), dtype=float)
        FWHMs = np.asarray(f.readline().rstrip().rstrip(',\"').replace('FWHM,,\"', '').split('\",\"'), dtype=float)
        areas = np.asarray(f.readline().rstrip().rstrip(',\"').replace('Area,,\"', '').split('\",\"'), dtype=float)
        lineshapes = f.readline().rstrip().rstrip(',\"').replace('Lineshape,,\"', '').split('\",\"')
    
    # Create a dictionary with metadata
    fit_summary = {
        'region' : region,
        'acq_energy' : acq_energy,
        'names' : names,
        'centers' : centers,
        'FWHMs' : FWHMs,
        'areas' : areas,
        'lineshapes' : lineshapes
    }
    
    # Read the rest of the data
    data_df = pd.read_csv(file_path, sep=',', skiprows=7, index_col='B.E.', usecols=range(1,6+len(centers))).dropna(axis=1)
    
    return fit_summary, data_df

def plot_spectrum(filename, fit_summary, data_df, output_dir=None, size=7, aspect_ratio=1.5,
                  subtract_background=False, display_residuals=True, normalize=True):
    """
    Create and display individual visualizations of the spectral data.
    
    Parameters:
    filename : str
        String containing the file name, without suffix
    fit_summary : dict
        Dictionary with metadata from the narrow scan
    data_df : pandas.DataFrame
        DataFrame containing the intensity data with binding energies as index and data, components, background and envelope as columns
    output_dir : str or Path, optional
        Directory for saving plots. If None, plots are only displayed inline
    size : int
        Specifies the overall size of the picture
    aspect_ratio : float
        Aspect ratio of the final picture
    subtract_background : bool
        Whether to subtract the background in the plot
    display_residuals : bool
        Whether to display residuals at the bottom of the plot
    normalize : bool
        Whether to normalize intensity values to [0,1]
    """
    # Normalize data if requested
    if normalize:
        plot_data = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())
    else:
        plot_data = data_df
    
    # Retrieve fit window from data, to plot vertical limits and to avoid ugly representation of components out of the fit window of CasaXPS
    mask = data_df['Background'] != data_df['Counts']
    left_limit = mask.idxmax()
    right_limit = mask.iloc[::-1].idxmax()
    
    # Create the plot with the specified aspect ratio
    fig, ax = plt.subplots(figsize=(size*aspect_ratio,size))

    # Plot counts, components and envelope subtracting the background, if requested
    if subtract_background:
        plt.plot(plot_data['Counts']-plot_data['Background'], marker='o', ms=2,  c='k', ls='')
        for component in fit_summary['names']:
            plt.plot(plot_data[component]-plot_data['Background'], label=r'{}'.format(component))
        if len(fit_summary['names']) != 1:
            plt.plot(plot_data['Envelope']-plot_data['Background'], lw=2, c='silver')

        # Display the residuals of the fit, if requested
        if display_residuals:
            plt.hlines(-0.2, left_limit, right_limit, colors='grey', linewidths=1)
            residuals = plot_data.loc[left_limit:right_limit,'Counts']-plot_data.loc[left_limit:right_limit,'Envelope']
            plt.plot(0.5*(residuals)-0.2, lw=1, c='tan')
            # plt.annotate('Residuals', (0.5*(right_limit+left_limit), 0.5*residuals.max()-0.2), color='grey', ha='center', va='bottom')
    
    # Otherwise, plot without background subtraction
    else:
        plt.plot(plot_data['Counts'], marker='o', ms=2,  c='k', ls='')
        plt.plot(plot_data.loc[left_limit:right_limit,'Background'], lw=1, c='lightcoral')
        for component in fit_summary['names']:
            plt.plot(plot_data.loc[left_limit:right_limit,component], label=r'{}'.format(component))
        if len(fit_summary['names']) != 1:
            plt.plot(plot_data.loc[left_limit:right_limit,'Envelope'], lw=2, c='silver')

        # Display the residuals of the fit, if requested
        if display_residuals:
            plt.hlines(-0.1, left_limit, right_limit, colors='grey', linewidths=1)
            residuals = plot_data.loc[left_limit:right_limit,'Counts']-plot_data.loc[left_limit:right_limit,'Envelope']
            plt.plot(0.5*residuals-0.1, lw=1, c='tan')
            # plt.annotate('Residuals', (0.5*(right_limit+left_limit), 0.5*residuals.max()-0.1), color='grey', ha='center', va='bottom')

    # Plot vertical lines to mark the fit window
    lower_limit, upper_limit = ax.get_ylim()
    plt.vlines([left_limit,right_limit], upper_limit, lower_limit, colors='grey', linestyles='dashed', linewidths=1)
    
    # Set other features of the plot, including axes and legend
    ax.legend()
    ax.set_title(fit_summary['region'])
    ax.set_xlabel(r'E$_B$ (eV)')
    ax.minorticks_on()
    ax.invert_xaxis()
    
    # Adjustments to y axis
    if normalize:
        ax.set_yticks(ax.get_yticks(), labels=[])
        ax.set_ylim(bottom=lower_limit, top=upper_limit)
        ax.set_ylabel('Intensity (arb. u.)')
    else:
        ax.set_ylim(bottom=lower_limit, top=upper_limit)
        ax.set_ylabel('Counts')
    
    # Save if output directory is provided
    if output_dir:
        output_path = os.path.join(output_dir, f'{filename}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_path}")
    
    # Show plot (for running as script)
    plt.show()
    plt.close(fig)
    
    return

def process_single_spectra(data_dir=None, output_dir=None, store_data=False, size=7, aspect_ratio=1.5,
                          subtract_background=False, display_residuals=True, normalize=True):
    """
    Individually plot all spectral data files in the specified directory.
    
    Parameters:
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to current directory.
    output_dir : str or Path, optional
        Directory for saving plots. Defaults to None (plots only displayed inline)
    store_data : bool
        Whether to archive data in a dictionary returned by the function
    size : int
        Specifies the overall size of the picture
    aspect_ratio : float
        Aspect ratio of the final pictures
    subtract_background : bool
        Whether to subtract the background in the plot
    display_residuals : bool
        Whether to display residuals at the bottom of the plot
    normalize : bool, optional
        Whether to normalize intensity values
        
    Returns:
    dict: Dictionary mapping file names to their processed DataFrames
    """
    # Set directories up
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Get all csv files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("No .csv files found in the specified directory.")
        return {}
    
    print(f"Processing {len(csv_files)} files...")
    
    # Process each file
    processed_data = {}
    for file_path in tqdm(csv_files, desc="Processing spectra"):
        try:
            # Read and process the single file
            fit_summary, data_df = read_spectral_data(file_path)
            
            # Store processed data
            if store_data:
                processed_data[file_path.stem] = {
                    'summary': fit_summary,
                    'data': data_df
                }
            
            # Create and display plots
            plot_spectrum(
                file_path.stem,
                fit_summary,
                data_df,
                output_dir=output_dir,
                size=size,
                aspect_ratio=aspect_ratio,
                subtract_background=subtract_background,
                display_residuals=display_residuals,
                normalize=normalize
            )
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return processed_data

# Main execution block (only runs when script is executed directly)
if __name__ == "__main__":
    # Example usage
    results = process_single_spectra(
        data_dir='spectra_to_process',
        output_dir='plots',
        store_data=True,
        size=4,
        # aspect_ratio=2,
        # subtract_background=True,
        # display_residuals=False,
        # normalize=False,
    )