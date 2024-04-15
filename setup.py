from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def setup_package():
    """
    Set up the Benchmarking package.

    This function reads the requirements from a file, sets up the package with the specified name and version,
    and installs the required dependencies.

    """
    setup(name='Detecting-Errors-through-Ensemblings-Prompts', version='1.0', packages=find_packages(), install_requires=requirements)

setup_package()

