from setuptools import find_packages, setup

exec(open("pyknos/version.py").read())

setup(
    name="pyknos",
    version=__version__,
    description="",
    url="https://github.com/mackelab/pyknos",
    author="Conor Durkan, George Papamakarios, Artur Bekasov",
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
