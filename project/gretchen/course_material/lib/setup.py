from setuptools import setup, find_packages
import platform

setup(
    name='gretchen',
    version='0.4',
    packages=['gretchen'],
    package_dir={'': 'src'},
    license='Apache 2.0',
    description='Gretchen robot & camera driver',
    #install_requires=['pyserial']
)
