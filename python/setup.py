#!/usr/bin/env python

from setuptools import setup, find_packages
import io
import os

# To publish to pypi, run:
# rm -rf ./build ./dist opengen.egg-info ; pip install . ; python setup.py sdist bdist_wheel; twine upload dist/*

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'gputils_api'

# Import version from file
version_file = open(os.path.join(here, 'VERSION'))
VERSION = version_file.read().strip()

DESCRIPTION = 'Python API for GPUtils'


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=['Pantelis Sopasakis', 'Ruairi Moran'],
      author_email='xzu.trustful191@passinbox.com',
      license='GNU General Public License v3 (GPLv3)',
      packages=find_packages(
          exclude=["private"]),
      include_package_data=True,
      install_requires=[
          'numpy', 'setuptools'
      ],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python',
          'Environment :: GPU :: NVIDIA CUDA',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Libraries'
      ],
      keywords=['api', 'GPU'],
      url=(
          'https://github.com/GPUEngineering/GPUtils'
      ),
      zip_safe=False)
