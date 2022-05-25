""" Setup for OpArt library."""

import codecs
import os
import re

import setuptools


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "src", "__init__.py")


def get_requirements():
    """Buffer required dependencies."""
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_version():
    """Get version number."""
    with open(VERSION_FILE, "rt", encoding="utf-8") as file:
        lines = file.readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        version = re.search(version_regex, line, re.M)
        if version_regex:
            return version.group(1)
    raise RuntimeError(f"Unable to find version in {VERSION_FILE}.")


setuptools.setup(
    name="src",
    version=get_version(),
    description="",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    long_description_content_type="text/markdown",
    include_package_data=True,
)
