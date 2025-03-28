from setuptools import setup, find_packages

setup(
    name="house_prices",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "jupyter>=1.0.0",
        "flake8>=6.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
) 