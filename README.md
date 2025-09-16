# xps

CNI-IIT's repository of XPS data analysis and plotting procedures.

## Features

### Currently available

- Plot data and components elaborated via *CasaXPS* (`.ipynb` version)

### To be added

- Plot data and components elaborated via *CasaXPS* (`.py` version)

### Under development :construction:

- Data import and fitting (`Specs-xy-data` folder)
  - `xy-data-import.py`: reads every .xy file in the folder and outputs it as .csv file.
  - `xy-data-fit.py`: gets .yaml file for fitting parameters (can be done with a dict directly - see example section) and .csv data, outputs in a desired folder fit report (.txt), fit data (.csv), and images (.png, .svg and .eps).
  - ...

## Running the code

The provided code is intended to be executed via our **Standard execution toolbox**, i.e.:

- via *Jupyter Lab* installed and launched through *miniconda3*, for any `.ipynb` files (jupyter notebooks), or
- via [*VS Code*](https://code.visualstudio.com/download) endowed with *Python* and/or *Jupyter* extensions, for any `.py` script and/or `.ipynb` files, respectively.

Please, refer to the organization's guidelines for further details.

## Installation

See installation guide (`INSTALL.md`).

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
