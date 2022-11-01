import setuptools
setuptools.setup(
     name='causal_world',
     version='1.2',
     author="Ossama Ahmed, Frederik Trauble",
     author_email="ossama.ahmed@mail.mcgill.ca, frederik.traeuble@outlook.com",
     description="CausalWorld: A Robotic Manipulation Benchmark "
                 "for Causal Structure and Transfer Learning",
     url="https://github.com/rr-learning/CausalWorld",
     packages=setuptools.find_packages(),
     package_data={'causal_world': ['assets/baseline_actors/*.zip',
                                    'assets/robot_properties_fingers/meshes/stl/edu/*.stl',
                                    'assets/robot_properties_fingers/meshes/stl/*.stl',
                                    'assets/robot_properties_fingers/urdf/*.urdf']},
     include_package_data=True,
     install_requires=[
        'pybullet==3.2.5',
        'gym==0.17.2',
        'numpy',
        'catkin_pkg',
        'sphinx',
        'matplotlib',
        'sphinx_rtd_theme',
        'sphinxcontrib-bibtex',
        'pytest',
        'psutil',
        'scipy'
      ],
    zip_safe=False
    )
