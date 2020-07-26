from setuptools import find_packages, setup

exec(open("nflows/version.py").read())

setup(
    name="nflows",
    version=__version__,
    description="Normalizing flows in PyTorch.",
    url="https://github.com/bayesiains/nflows",
    download_url = 'https://github.com/bayesiains/nflows/archive/v0.12.tar.gz',
    author="Conor Durkan, Artur Bekasov, George Papamakarios, Iain Murray",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
        "dev": [
            "autoflake",
            "black",
            "flake8",
            "isort",
            "pytest",
            "pyyaml",
            "torchtestcase",
        ],
    },
    dependency_links=[],
)
