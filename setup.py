#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lya power spectrum routines"
version = "1.0"

setup(
    name="forestflow",
    version=version,
    description=description,
    url="https://github.com/igmhub/ForestFlow",
    author="Jonas Chaves-Montero, Laura Cabayol-Garcia",
    author_email="jchaves@ifae.es",
    packages=["forestflow"],
    install_requires=["numpy", "pydoe2", "emcee", "FrEIA", "corner", "jupyter"],
    zip_safe=False,
)
