import importlib
import pytest


def test_package_import(package_name):
    package_found = importlib.util.find_spec(package_name) is not None
    assert package_found, f"Package {package_name} should be installed."


def test_packages():
    packages = ["housing_prediction"]
    for package in packages:
        test_package_import(package)
