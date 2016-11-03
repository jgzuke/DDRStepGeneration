from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'GetMusicTrainingDataX',
  ext_modules = cythonize("GetMusicTrainingDataX.pyx"),
)
