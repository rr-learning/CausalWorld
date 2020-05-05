from setuptools import setup
from setuptools import find_packages
from os.path import splitext
from glob import glob
from os.path import basename

setup(name='causal_rl_bench',
      version='1.0.0',
      packages=find_packages('python/src'),
      package_dir={'': 'python/src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
)