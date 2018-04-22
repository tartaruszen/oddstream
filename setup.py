from setuptools import setup
from setuptools import find_packages

setup(name = 'oddstream', 
      version = '0.1',
      description = 'Fill this',
      url = 'http://github.com/anofox/oddstream',
      author = 'Simon Müller',
      author_email = 'sm@anofox.com',
      licence = 'MIT',
      install_requires = ['scikit-learn',
                          'numpy',
                          'fastkde'],
      zip_safe = False,
      packages=find_packages())
 
