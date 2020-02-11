from setuptools import find_packages, setup

exec(open("sbi/version.py").read())

setup(
    name="sbi",
    version=__version__,
    description="Simulation-based inference",
    url="https://github.com/mackelab/sbi",
    author="Conor Durkan",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    test_requires=["pytest", "deepdiff", "torchtestcase"],
    install_requires=[
        "matplotlib",
        "numpy",
        "pyro-ppl",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={"dev": ["autoflake", "black", "flake8", "isort", "pytest"]},
    dependency_links=[],
)
