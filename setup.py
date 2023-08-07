"""
Setup file for package `fargo_helper`.
"""
import setuptools
import pathlib

PACKAGENAME = 'fargo_helper'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).absolute().parent


def read_version():
    "function to parse the version from"
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":

    setuptools.setup(
        name=PACKAGENAME,
        description='helper functions to read fargo3d output',
        version=read_version(),
        long_description=(HERE / "README.md").read_text(),
        long_description_content_type='text/markdown',
        url='https://github.com/birnstiel/fargo_helper',
        author='Til Birnstiel',
        author_email='til.birnstiel@lmu.de',
        license='GPLv3',
        packages=setuptools.find_packages(),
        include_package_data=True,
        install_requires=['matplotlib', 'numpy'],
        python_requires='>=3.6',
        zip_safe=False
    )
