from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = 'TDMA solver',
      ext_modules = cythonize("tdma.pyx"),
)
