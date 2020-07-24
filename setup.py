from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["causal_rl_bench"],
    package_dir={"": "python/src"},
    package_data={},
)

setup(**d)
