from setuptools import setup, find_packages

setup(name='PPO',
      version='0.1',
      install_requires=['numpy', 'torch'],
      packages=find_packages(),
      )