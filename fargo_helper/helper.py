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


def get_numbers(outputdir):
    "get all the string numbers of the snapshots in the folder"
    outputdir = Path(outputdir)
    N_str = [d.stem.replace('gasdens', '') for d in outputdir.glob('gasdens*') if '2d' not in d.name]
    N = [int(_N) for _N in N_str]
    idx = np.argsort(N)
    N_str = [N_str[i] for i in idx]
    return N_str


def read_planet_files(outputdir, time=None, big=False):
    """Reads planet data from the planet files

    Parameters
    ----------
    outputdir : path
        path where the output data can be found
    time : None | flot
        if None, returns all data, if a string gives the closest snapshot

    big : bool
        if True, read from bigplanet files instead of planet files

    Returns
    -------
    dictionary
        one entry for each planet, containing 
        snapshot,x,y,z,vx,vy,vz,mass,time,omegaframe
    """
    stem = 'big' * big + 'planet'
    files = Path(outputdir).glob(stem + '*.dat')
    planets = {}
    for file in files:
        ip = int(file.stem.split(stem)[1])
        d = np.loadtxt(file)

        if time is not None:
            it = np.abs(d[:, -2] - time).argmin()
            d = d[it, :]

        planets[ip] = d
    return planets


def read_fargo(outputdir, N, dtype=None, keys='dens', verbose=False):
    """read fargo output

    Reads the domain (in x, y, z) and the given keys from output #N.

    Keys can be a comma separated float. 'all' will be
    interpreted as 'rho,vx,vy,vz'

    Tries to read time from summary.

    Returns a `SimpleNamespace` with all data.
    """

    N = int(float(N))

    if keys == 'all':
        keys = 'dens,vx,vy,vz'

    keys = [k.strip() for k in keys.split(',')]

    out = SimpleNamespace()
    out.outputdir = Path(outputdir)
    out.N = N
    out.Ns = get_numbers(out.outputdir)

    # update snapshot number
    if N == -1:
        # find maximum snapshot number
        out.N = int(out.Ns[out.N])

    # try to read summary and get the time from it
    try:
        out.summary = next(out.outputdir.glob(f'summary{out.N:d}.dat')).read_text()
        out.time = float(out.summary.split('at simulation time ')[1].split(' ')[0])
    except Exception:
        out.time = None
        out.summary = None

    # try and read the parameters
    try:
        out.params = Parameters(out.outputdir)
    except Exception:
        out.params = None

    # try and read the planet properties
    out.planets = read_planet_files(out.outputdir, time=out.time)

    # try and find out the data type (single or double)
    if dtype is None:
        if hasattr(out.params, 'realtype'):
            out.dtype = getattr(np, out.params.realtype.strip('"'))
        else:
            out.dtype = np.float64

    # read the grid
    if out.params.coordinates == 'spherical':

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
    elif out.params.coordinates == 'cylindrical':

        if (out.outputdir / 'domain_x.dat').is_file():
            out.phii = np.loadtxt(out.outputdir / 'domain_x.dat')
            out.phi = 0.5 * (out.phii[1:] + out.phii[:-1])
            out.nphi = len(out.phi)

        if (out.outputdir / 'domain_y.dat').is_file():
            out.ri = np.loadtxt(out.outputdir / 'domain_y.dat')[3:-3]
            out.r = 0.5 * (out.ri[1:] + out.ri[:-1])
            out.nr = len(out.r)

    else:
        raise ValueError('only spherical coordinates are implemented')

    # handle gas quantities

    for q in keys:
        try:
            res = np.fromfile(out.outputdir / f'gas{q}{out.N}.dat', dtype=out.dtype)
            if out.params.coordinates == 'spherical':
                setattr(out, q, res.reshape(out.nth, out.nr, out.nphi).transpose(1, 2, 0))
            elif out.params.coordinates == 'cylindrical':
                setattr(out, q, res.reshape(out.nr, out.nphi))
            del res
        except FileNotFoundError as e:
            if verbose:
                print(e)

    # handle dust quantities

    invSt_keys = [key for key in out.params.__dict__.keys() if key.startswith('invstokes')]
    n_dust = len(invSt_keys)

    for i_dust in range(1, n_dust + 1):
        for q in keys:
            try:
                res = np.fromfile(out.outputdir / f'dust{i_dust}{q}{out.N}.dat', dtype=out.dtype)
                if out.params.coordinates == 'spherical':
                    setattr(out, f'dust{i_dust}{q}', res.reshape(out.nth, out.nr, out.nphi).transpose(1, 2, 0))
                elif out.params.coordinates == 'cylindrical':
                    setattr(out, f'dust{i_dust}{q}', res.reshape(out.nr, out.nphi))
                del res
            except FileNotFoundError as e:
                if verbose:
                    print(e)

    out.rho = out.dens
    out.n_dust = n_dust

    return out
