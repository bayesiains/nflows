from setuptools import find_packages, setup

exec(open("nflows/version.py").read())

setup(
    name="nflows",
    version=__version__,
    description="",
    url="https://github.com/mackelab/nflows",
    author="Conor Durkan, Artur Bekasov, George Papamakarios",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    install_requires=[
        "matplotlib",
        "numpy",
        "pyro-ppl",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
        "dev": [
            "autoflake",
            "black",
            "deepdiff",
            "flake8",
            "isort",
            "pytest",
            "pyyaml",
            "torchtestcase",
        ],
    },
    dependency_links=[],
)
