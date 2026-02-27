"""
SAMpy: A Fourier-Plane Pipeline for NIRISS AMI Data (and more!)

See Sallum & Eisner (2017) and Sallum et al. (2022) for descriptions
of the pipeline.
"""

from importlib.resources import files as _resource_files


def get_data_path(filename):
    """Return the full path to a bundled data file.

    Parameters
    ----------
    filename : str
        Name of the file in the sampy/data/ directory,
        e.g. ``'NIRISS_7holeMask.txt'``.

    Returns
    -------
    pathlib.Path
        Absolute path to the data file.
    """
    return _resource_files("sampy.data").joinpath(filename)
