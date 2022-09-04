from os import path
from setuptools import find_packages, setup
from nflows.version import VERSION

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nflows",
    version=VERSION,
    description="Normalizing flows in PyTorch.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/bayesiains/nflows",
    download_url = 'https://github.com/bayesiains/nflows/archive/v0.14.tar.gz',
    author="Conor Durkan, Artur Bekasov, George Papamakarios, Iain Murray",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
        "umnn"
    ],
    dependency_links=[],
)
