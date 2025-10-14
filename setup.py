"""
Setup configuration for uplift-learn package
"""
from setuptools import setup, find_packages
from pathlib import Path

# README'yi oku
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="uplift-learn",
    version="0.1.0",
    author="Sumeyra Guclu",
    author_email="sumeyraguclu29@gmail.com",
    description="A hands-on learning project for uplift modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sumeyraguclu/uplift-learn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.2",
        "pandas>=2.1.4",
        "scipy>=1.11.4",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "scikit-uplift>=0.5.1",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "ruff>=0.1.9",
            "mypy>=1.7.1",
        ],
        "optimization": [
            "ortools>=9.8.3296",
        ],
    },
    entry_points={
        "console_scripts": [
            # Gün 6-7'de eklenecek CLI komutları
            # "uplift-train=src.cli:train",
            # "uplift-predict=src.cli:predict",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)