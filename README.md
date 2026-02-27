# SAMpy

A Fourier-Plane Pipeline for NIRISS AMI Data

See [Sallum & Eisner 2017](https://ui.adsabs.harvard.edu/abs/2017ApJS..233....9S/abstract) and [Sallum et al. 2022](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..2MS/abstract) for descriptions of the pipeline.

## Installation

Clone the repository and enter the directory:

```
git clone https://github.com/jordan-stone/SAMpy.git
cd SAMpy
```

Option 1 (recommended): use conda to create a new environment:

```
conda env create --file environment.yml
conda activate SAMpy
```

Option 2: install in editable mode with pip:

```
pip install -e ".[notebook]"
```

## Quick start

After installation, navigate to `examples/` and open the example notebook:

```
cd examples
jupyter notebook notebooks/ABDor_closure_phases.ipynb
```

## Package structure

```
sampy/               # Installable Python package
├── mask.py          # Aperture mask geometry and coordinate setup
├── reduction.py     # Image reading, cleaning, windowing
├── analysis.py      # Closure phases, visibilities, covariances
├── calibration.py   # Polynomial calibration
├── oi.py            # OIFITS file generation
├── utils.py         # Image centering and Fourier utilities
├── fitting/         # Model fitting subpackage
│   ├── mcmc.py      # emcee parallel-tempered MCMC
│   └── grid.py      # Chi-squared grid search
└── data/            # Bundled filter and mask files

examples/            # Demonstration material (not installed)
├── notebooks/       # Jupyter notebooks
├── scripts/         # Standalone utility scripts
└── data/            # Example FITS data
```

## Fork maintainer

* Jordan Stone

## Original authors

* Josh Eisner
* Kenzie Lach
* Shrishmoy Ray
* Steph Sallum
* Christina Vides

## License

GPL-3.0
