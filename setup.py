from setuptools import find_packages, setup

exec(open("lfi/version.py").read())

setup(
    name="lfi",
    version=__version__,
    description="LFI + CDE.",
    url="https://github.com/mackelab/lfi",
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
