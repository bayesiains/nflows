from setuptools import find_packages, setup

exec(open("pyknos/version.py").read())

setup(
    name="pyknos",
    version=__version__,
    description="",
    url="https://github.com/mackelab/pyknos",
    author="Conor Durkan, George Papamakarios, Artur Bekasov",
    packages=find_packages(exclude=["sbi", "tests"]),
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
