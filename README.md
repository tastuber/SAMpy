# SAMpy

A Fourier-plane aperture masking interferometry pipeline for extracting closure phases, squared visibilities, and complex visibilities from non-redundant mask (NRM) data. Originally developed for JWST/NIRISS AMI, SAMpy now supports custom instruments with arbitrary hole counts and either hexagonal or circular subapertures.

See [Sallum & Eisner 2017](https://ui.adsabs.harvard.edu/abs/2017ApJS..233....9S/abstract) and [Sallum et al. 2022](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..2MS/abstract) for descriptions of the pipeline.

## Installation

All installation options begin by cloning the repository and entering the directory:

```bash
git clone https://github.com/jordan-stone/SAMpy.git
cd SAMpy
```
Option 1 (recommended): use conda to create a new environment with SAMpy installed:

```bash
conda env create --file environment.yml
conda activate SAMpy
```
Option 2: install in editable mode with pip:

```bash
pip install -e ".[notebook]"   # with Jupyter support
pip install -e .               # core only
```

## Pipeline overview

SAMpy processes aperture masking interferometry data through six stages:

1. **Mask setup** (`mask.make_coords`) — Compute Fourier-plane sampling coordinates for the aperture mask geometry. Supports any non-redundant mask with hexagonal or circular subapertures. JWST/NIRISS mask and filter files are bundled for convenience. Outputs are cached as FITS files; recomputation is only needed when the mask geometry *or* photometric filter changes. This step takes ~30 minutes but only needs to run once per mask + filter combination.

2. **Reduction** (`reduction`) — Read pipeline products, fix bad pixels, subtract backgrounds, subframe, and apply a super-Gaussian apodization window.

3. **Observable extraction** (`analysis`) — Extract closure phases, squared visibilities, and complex visibilities from the image stack. Each returns a dictionary with the observables, covariance matrices, and per-image scatter.

4. **Calibration** (`calibration`) — Remove instrumental closure phase and visibility biases using polynomial calibration against reference stars observed at different times.

5. **OIFITS export** (`oi`) — Package calibrated observables into the standard OIFITS format for interoperability with other interferometry tools.

6. **Model fitting** (`fitting`) — Fit binary companion models via chi-squared grid search or parallel-tempered MCMC.

## Quick start

After installation, open the example notebook:

```bash
cd examples
jupyter notebook notebooks/ABDor_closure_phases.ipynb
```

The notebook walks through the full pipeline on JWST/NIRISS commissioning data of AB Doradus: image reduction, Fourier observable extraction, calibration, OIFITS generation, and companion fitting.

## Key API reference

### Mask setup (`mask`)

#### JWST/NIRISS (built-in shortcut)

NIRISS mask and filter data are bundled with SAMpy, so you only need to specify a filter name. The built-in 7-hole mask coordinates (`NIRISS_7holeMask.txt`, flat-to-flat diameter 0.75 m) and four filter curves (`f277w`, `f380m`, `f430m`, `f480m`) are used automatically:

```python
from sampy.mask import make_coords

# First run (or when filter changes) — takes ~30 min
make_coords('f480m_niriss_mask/', jwst_filt='f480m', recompute=True)

# Subsequent runs — uses cached files
make_coords('f480m_niriss_mask/', jwst_filt='f480m')
```

#### Custom instruments

For any other instrument, provide three things: a mask hole coordinate file, a subaperture diameter, and a filter transmission curve. These must all be specified together — omitting any one raises an error immediately.

```python
make_coords(
    'lbti_Lband_mask/',
    mask_file='lbti_holes.txt',       # x,y coords in meters, one row per hole
    subaperture_diameter=0.5,         # meters (diameter for circular, flat-to-flat for hex)
    filter_file='Lband_filter.txt',   # wavelength (µm) vs throughput
    hole_shape='circular',            # 'hexagonal' or 'circular'
    pixel_scale=0.0107,               # arcsec/pixel
    pupil_pixel_scale=0.006,          # finer sampling for smaller subapertures (default 0.025)
    n_pixels=512,
    recompute=True,
)
```

**File formats:**

- **Mask file** — whitespace-delimited text, one row per hole, two columns: x and y position in meters. No header line. (See `sampy/data/NIRISS_7holeMask.txt` for an example.)
- **Filter file** — whitespace-delimited text with one header line, wavelength in microns in column 0, throughput in column 1. Additional columns are ignored. (See `sampy/data/NIRISS_F380M.txt` for an example.)

**Key parameters:**

- `pupil_pixel_scale` controls the pupil-plane model resolution in meters per pixel. Smaller values sample the subapertures more finely at higher computational cost. The default (0.025) gives ~30 pixels across each NIRISS subaperture; instruments with smaller subapertures may need a smaller value.
- `zero_spacing_radius` sets the radius (in pixels) of the exclusion zone around the zero-spacing (DC) peak in the power spectrum. You may need to tune this for your instrument geometry.

Each mask directory is specific to a mask geometry + filter pair. If you change the filter but keep the same mask, you need a new output directory (or `recompute=True`).

### Observable extraction (`analysis`)

The main functions return dictionaries with descriptive keys:

```python
from sampy.analysis import calc_cps_multi, calc_v2s

# Closure phases
cps = calc_cps_multi(images, mask_dir)
cps['closure_phases']   # mean closure phases (degrees), shape (n_triangles,)
cps['covariance']       # covariance matrix
cps['triple_amps']      # mean triple amplitudes
cps['raw']              # per-image bispectra (complex)

# Squared visibilities
v2s = calc_v2s(images, mask_dir, save_allpix=True, filename='output_base')
v2s['v2']               # mean bias-corrected V², shape (n_baselines,)
v2s['covariance']       # covariance matrix
v2s['v2_scatter']       # per-image bias-corrected V²
```

Variants include `calc_cps_single`, `calc_cps_pavg`, `calc_cps_multi_DFT`, `calc_v2s_single`, `calc_cvis`, and `_groups` versions for grouped/cubed data.

### Calibration (`calibration`)

```python
from sampy.calibration import polynomial_calibrate

# Calibrate closure phases against reference stars
result = polynomial_calibrate(target_cps, cal_cps, target_times,
                              cal_times, poly_order, 'cps')
```

### Model fitting (`fitting`)

```python
from sampy.fitting.grid import generate_chi2_grid, process_binary_grid
from sampy.fitting.mcmc import run_pt_emcee, plot_corner

# Grid search
grid, coords = generate_chi2_grid(cp_data, cp_errs, cp_uvs,
                                  v2_data, v2_errs, v2_uvs,
                                  angles, wavelength)
best, min_chi2 = process_binary_grid(grid, coords, cp_data, v2_data)

# MCMC refinement
chain, lnp, lnl = run_pt_emcee(v2_data, v2_errs, v2_uvs,
                                cp_data, cp_errs, cp_uvs,
                                angles, wavelength, include_v2s=True)
best_params, medians = plot_corner(chain, lnl,
                                   degrees_of_freedom=n_cp + n_v2 - 3)
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
└── data/            # Bundled NIRISS mask and filter files

examples/            # Demonstration material (not installed)
├── notebooks/       # Jupyter notebooks
├── scripts/         # Standalone utility scripts
└── data/            # Example FITS data
```

## Dependencies

SAMpy requires Python 3.9–3.11 and the following key packages:

- **numpy** (>=1.22), **scipy**, **matplotlib**, **astropy**
- **emcee** 2.x (<3.0) — the parallel-tempered sampler (`PTSampler`) was removed in emcee 3.x
- **oifits** (>=0.4) — Paul Boley's [OIFITS reader/writer](https://github.com/pboley/oifits)
- **shapely** — for hexagonal subaperture geometry (not needed for circular holes)
- **jwst** — for reading JWST pipeline products (only needed for JWST data)

See `pyproject.toml` for the full dependency list.

## Migrating from earlier versions

### Return type changes

`calc_cps_multi`, `calc_cps_pavg`, `calc_cps_single`, `calc_cps_single_DFT`, and `calc_cps_multi_DFT` previously returned `np.array([...], dtype=object)` accessed by integer index. They now return dictionaries:

| Old index | New key            | Description                    |
|-----------|--------------------|--------------------------------|
| `[0]`     | `'raw'`            | Per-image bispectra (complex)  |
| `[1]`     | `'closure_phases'` | Mean closure phases (degrees)  |
| `[2]`     | `'triple_amps'`    | Mean triple amplitudes         |
| `[3]`     | `'covariance'`     | Covariance matrix              |
| `[4]`     | `'variance'`       | Variance (diagonal)            |
| `[5]`     | `'std_error'`      | Standard error of the mean     |

`calc_v2s`, `calc_v2s_single`, and `calc_v2s_groups`:

| Old index | New key          | Description                       |
|-----------|------------------|-----------------------------------|
| `[0]`     | `'v2'`           | Mean bias-corrected V²            |
| `[1]`     | `'covariance'`   | Covariance matrix                 |
| `[2]`     | `'variance'`     | Variance                          |
| `[3]`     | `'std_error'`    | Standard error                    |
| `[4]`     | `'v2_scatter'`   | Per-image bias-corrected V²       |
| `[5]`     | `'amplitudes'`   | Zero-frequency power per image    |
| `[6]`     | `'unnormalized'` | Unnormalized summed power         |
| `[7]`     | `'bias'`         | Bias estimates per image          |

`calc_cvis` and `calc_cvis_groups` follow a similar pattern with keys `'amplitudes'`, `'phases'`, `'covariance'`, `'variance'`, `'std_error'`, `'phases_per_image'`.

### Keyword renames

| Module         | Old name        | New name                |
|----------------|-----------------|-------------------------|
| `analysis`     | `useW=`         | `use_weights=`          |
| `mask`         | `redo=`         | `recompute=`            |
| `mask`         | `mask_radius=`  | `zero_spacing_radius=`  |
| `reduction`    | `half_width_hm=`| `fwhm=`                 |

### Behavioral changes

- **`calc_v2s` and `calc_v2s_groups`**: `save_allpix` default changed from `True` to `False`. If your scripts relied on HDF5 output without explicitly passing `save_allpix=True`, add it.
- **`plot_corner`**: `degrees_of_freedom` is now required (was `np.nan`). Pass `n_data_points - n_model_params`, e.g. `degrees_of_freedom=n_cp + n_v2 - 3`.
- **`oifits`**: The bundled `sampy/oifits.py` has been removed in favor of the [PyPI package](https://github.com/pboley/oifits), which is now installed automatically as a dependency. Change `from sampy.oifits import ...` to `from oifits import ...` in your scripts.

## Fork maintainer

- Jordan Stone

## Original authors

- Josh Eisner
- Kenzie Lach
- Shrishmoy Ray
- Steph Sallum
- Christina Vides

## License

GPL-3.0
