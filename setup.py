#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()


requirements = ["flammkuchen", "numpy"]

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()

setup(
    author="Vilim Stih & Luigi Petrucco @portugueslab",
    author_email="luigi.petrucco@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A package for HDF5-based chunked arrays",
    install_requires=requirements,
    extras_require=dict(dev=requirements_dev),
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="split_dataset",
    name="split_dataset",
    packages=find_packages(include=["split_dataset", "split_dataset.*"]),
    test_suite="tests",
    url="https://github.com/portugueslab/split_dataset",
    version="0.4.2",
    zip_safe=False,
)
