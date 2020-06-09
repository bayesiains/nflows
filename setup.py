from setuptools import find_packages, setup

exec(open("nflows/version.py").read())

setup(
    name="nflows",
    version=__version__,
    description="",
    url="https://github.com/bayesiains/nflows",
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
