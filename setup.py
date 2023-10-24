"""Setup file for the asboostreg package."""
import setuptools

from model_helpers import __author__
from model_helpers import __email__
from model_helpers import __license__
from model_helpers import __maintainer__
from model_helpers import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

print("Installing the asboostreg package.")
print("Please check that the following dependencies are installed:")
print(*install_requires, sep="\n")

setuptools.setup(
    name="asboostreg",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="Additive Sparse Boosting Regression algorithm.",
    long_description=long_description,
    url="https://github.com/thesis-jdgs/additive-sparse-boost-regression",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    license=__license__,
    maintainer=__maintainer__,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    py_modules=["asboostreg"],
    include_package_data=True,
    package_data={"potts": ["*.dll"]},
)
