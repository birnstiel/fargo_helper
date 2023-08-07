from types import SimpleNamespace
from pathlib import Path
import numpy as np
from ast import literal_eval


class Parameters():
    """
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally variables.par
    """

    def __init__(self, directory=''):
        directory = Path(directory)
        try:
            params = open(directory / "variables.par", 'r')
        except IOError:  # Error checker.
            print("parameter file not found.")
            return

        lines = params.readlines()      # Reading the parfile
        params.close()                  # Closing the parfile
        par = {}                        # Allocating a dictionary
        for line in lines:              # Iterating over the parfile
            name, value = line.split()  # Spliting the name and the value (first blank)
            try:
                val = literal_eval(value)
            except Exception:
                val = value
            par[name] = val

        # A control atribute, actually not used, good for debbuging
        self._params = par
        for name in par:                # Iterating over the dictionary
            # Making the atributes at runtime
            setattr(self, name.lower(), par[name])


def read_fargo_2D(outputdir, N):
    """read fargo output

    Reads the domain (in x, y) and gas density.

    Tries to read time from summary.

    Returns a `SimpleNamespace` with all data.
    """

    outputdir = Path(outputdir)

    phii = np.loadtxt(outputdir / 'domain_x.dat')
    phi = 0.5 * (phii[1:] + phii[:-1])
    nphi = len(phi)

    ri = np.loadtxt(outputdir / 'domain_y.dat')[3:-3]
    r = 0.5 * (ri[1:] + ri[:-1])
    nr = len(r)

    sig = np.fromfile(outputdir / f'gasdens{N}.dat').reshape(nr, nphi)

    try:
        summary = next(outputdir.glob(f'summary{N:d}.dat')).read_text()
        time = float(summary.split('at simulation time ')[1].split(' ')[0])
    except Exception:
        time = None
        summary = None

    out = SimpleNamespace()
    out.time = time
    out.phi = phi
    out.phii = phii
    out.nphi = nphi
    out.r = r
    out.ri = ri
    out.nr = nr
    out.sig = sig
    out.summary = summary
    return out


def read_fargo(outputdir, N, dtype=None, keys='dens'):
    """read fargo output

    Reads the domain (in x, y, z) and the given keys.

    Keys can be a comma separated float. 'all' will be
    interpreted as 'rho,vx,vy,vz'

    Tries to read time from summary.

    Returns a `SimpleNamespace` with all data.
    """

    if keys == 'all':
        keys = 'dens,vx,vy,vz'

    keys = [k.strip() for k in keys.split(',')]

    out = SimpleNamespace()
    out.outputdir = Path(outputdir)

    # try to read summary and get the time from it
    try:
        out.summary = next(out.outputdir.glob(f'summary{N:d}.dat')).read_text()
        out.time = float(out.summary.split('at simulation time ')[1].split(' ')[0])
    except Exception:
        out.time = None
        out.summary = None

    # try and read the parameters
    try:
        out.params = Parameters(out.outputdir)
    except Exception:
        out.params = None

    # try and find out the data type (single or double)
    if dtype is None:
        if hasattr(out.params, 'realtype'):
            out.dtype = getattr(np, out.params.realtype.strip('"'))
        else:
            out.dtype = np.float64

    # read the grid
    if out.params.coordinates != 'spherical':
        raise ValueError('only spherical coordinates are implemented')

    if (out.outputdir / 'domain_x.dat').is_file():
        out.phii = np.loadtxt(out.outputdir / 'domain_x.dat')
        out.phi = 0.5 * (out.phii[1:] + out.phii[:-1])
        out.nphi = len(out.phi)

    if (out.outputdir / 'domain_y.dat').is_file():
        out.ri = np.loadtxt(out.outputdir / 'domain_y.dat')[3:-3]
        out.r = 0.5 * (out.ri[1:] + out.ri[:-1])
        out.nr = len(out.r)

    if (out.outputdir / 'domain_z.dat').is_file():
        out.thi = np.loadtxt(out.outputdir / 'domain_z.dat')[3:-3]
        out.th = 0.5 * (out.thi[1:] + out.thi[:-1])
        out.nth = len(out.th)

    for q in keys:
        res = np.fromfile(out.outputdir / f'gasdens{N}.dat', dtype=out.dtype)
        setattr(out, q, res.reshape(out.nth, out.nr, out.nphi).transpose(1, 2, 0))
        del res

    out.rho = out.dens

    return out
