from setuptools import setup, find_packages

setup(
    name="math-reasoner",
    version="0.1.0",
    description="A project exploring the use of language models for mathematical reasoning",
    author="Anas Lecaillon, Timothée Colette",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "tqdm>=4.64.0",
        "wandb>=0.13.0",
        "trl>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "ipython>=8.0.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 