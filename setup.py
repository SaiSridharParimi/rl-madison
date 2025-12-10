from setuptools import setup, find_packages

setup(
    name="madison-rl",
    version="0.1.0",
    description="Reinforcement Learning for Madison Agentic AI Framework",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.28.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)

