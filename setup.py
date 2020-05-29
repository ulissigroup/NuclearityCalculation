from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(name='drl_nuclearity_calc',
      version='0.0.1',
      description='Module for surface nuclearity calculation',
      url='https://github.com/ulissigroup/NuclearityCalculation',
      author='Unnatti Sharma, Zack Ulissi',
      author_email='zulissi@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=find_packages(),
      scripts=[],
      include_package_data=False,
      install_requires=['ase>=3.19.1',
			'numpy',
			'matplotlib',
			'pymatgen',
			'pandas',
			'gaspy @ git+https://github.com/ulissilab/GASpy.git'],
      long_description='''Module for surface nuclearity calculation for any slab''',)
