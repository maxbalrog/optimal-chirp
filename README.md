# optimal-chirp
Repository for different approaches for calculation of optimal chirp in laser pulse chirped in spectral domain. Here one can find 1) parameter scan on a linear grid, 2) analytic predictions, 3) more advanced numerical optimization (optuna).

**ComptonSpec_classic.py** - classes for classic calculation of Compton emission spectra taken from [here](https://github.com/maxbalrog/Compton_scattering_classic).

**generate_simulation_data.ipynb** - generate and save simulation data.

**Optuna_spectrum_optimization.ipynb** - numerical optimization of chirp with optuna package.

**plot_article_figures.ipynb** - plot all figures present in the article

**utils_simulation.py** - utilities for obtaining the simulation data from parameter scan on a linear $\beta$ grid.

**utils_analytics.py** - utilities for analytic predictions

Folder **data** - used simulation and analytic data (obtained with **generate_simulation_data.ipynb**).

Folder **Pearcey** - numerical calculation of Pearcey 2nd derivatives which were used in Taylor correction procedure.

This repository was created for the following article\
[1] - Valialshchikov M., Seipt D., Kharin V.Y., Rykovanov S.G. Towards high photon density for Compton Scattering by spectral
chirp. Phys. Rev. A 106 L031501 (2022). arXiv: https://arxiv.org/abs/2204.12245
