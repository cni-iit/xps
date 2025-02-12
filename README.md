# xps
CNI-IIT's repository of XPS data analysis and plotting procedures.

***

## Features

### Currently available
- Plot data and components elaborated via *CasaXPS* (`.ipynb` version)

### To be added
- Plot data and components elaborated via *CasaXPS* (`.py` version)

***

## Execution recommendations

The code available in the stable branch is certified to work with **Python 3.12.8**.  
Download a stable release of your choice and move the relevant code in your data folder, then run the code.

### Required packages
The required packages are listed below, with mutually compatible suggested versions:
- `numpy 2.2.1`
- `pandas 2.2.3`
- `matplotlib 3.10.0`
- `tqdm 4.66.5`
- `jupyterlab 4.3.4`
- `ipywidgets 8.1.5`

Begin the packages installation from `jupyterlab` to automatically install the right `python` version.  
Conversely, the packages from `jupyterlab` onwards are optional if the user intends to work with `.py` scripts only.

### Environment setup
To set up a proper environment, follow the **Standard environment creation procedure**:
1. Install [*miniconda3*](https://docs.anaconda.com/miniconda/install/),
2. Run the *Anaconda PowerShell Prompt*,
3. Input `conda create --name xps`, then `activate xps`,
4. Begin the packages installation with `conda install jupyterlab` and so on.

Please, refer to the organization's guidelines for further details.

### Running the code
The provided code is intended to be executed via our **Standard execution toolbox**, i.e.:
- via *Jupyter Lab* installed and launched through *miniconda3*, for any `.ipynb` files (jupyter notebooks), or
- via [*VS Code*](https://code.visualstudio.com/download) endowed with *Python* and/or *Jupyter* extensions, for any `.py` script and/or `.ipynb` files, respectively.

Please, refer to the organization's guidelines for further details.

***

## Development recommendations

Any help in further developing this repository is more than welcomed!  
To start contributing to our code, please employ our **Standard development toolbox**, i.e.:
- [*Git*](https://git-scm.com/downloads) as the source control manager,
- [*VS Code*](https://code.visualstudio.com/download) as the IDE, extended by:
  - *Python* and *Jupyter* support extensions,
  - *GitHub Pull Requests* extension.

Log in to GitHub in VS Code and clone this repository as a local repository on your machine, then branch it to work safely on the new features.  
When you're done, commit your edits and open a pull request.  
Also signing up for GitHub Copilot Free can be a good idea.
