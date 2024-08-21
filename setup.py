from setuptools import setup, find_packages

setup(name='PPO',
      version='0.1',
      install_requires=['numpy==1.26.1', 'torch==1.13.1', "gym==0.26.2", "tensorboard==2.14.1"],
      packages=find_packages(),
      )