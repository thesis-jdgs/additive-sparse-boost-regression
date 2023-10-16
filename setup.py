"""Setup file for the asboostreg package."""
import subprocess

import setuptools

from asboostreg import __version__
from model_helpers import __author__
from model_helpers import __email__
from model_helpers import __license__
from model_helpers import __maintainer__

installer_command = (
    "cd potts &&"
    " gcc -c -o l2_potts.o -fPIC -Ofast l2_potts.c  &&"
    " gcc -shared -o l2_potts.dll l2_potts.o"
)


class L2PottsInstallCommand(setuptools.Command):
    description = "Compile the C code for the L2-Potts."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(installer_command, shell=True)


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
    cmdclass={"install": L2PottsInstallCommand},
)
