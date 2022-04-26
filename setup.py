from setuptools import setup, find_packages
import os
import versioneer

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
install_requires = []
if os.path.isfile(path):
    with open(path) as f:
        install_requires = f.read().splitlines()

setup(
    name="mtj-softtuner",
    version=versioneer.get_version(),
    author="VE FORBRYDERNE",
    author_email="ve.forbryderne@gmail.com",
    url="https://github.com/ve-forbryderne/mtj-softtuner",
    packages=find_packages(),
    package_data={"": ["*.json", "*.py", "**/*.py", "kobold/maps/*.json"]},
    include_package_data=True,
    install_requires=install_requires,
    license="Apache",
    classifiers=[
        "Private :: Do Not Upload",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
    ],
)
