from setuptools import setup, find_packages

setup(name='metorch',
      version='0.1',
      description='A wrapper to train, test, and interpret ML models',
      author='Shashanka Subrahmanya',
      url='https://github.com/shshnk94/metorch',
      package_dir={'': 'src'},
      packages=find_packages('src', exclude=['experiments'])
     )
