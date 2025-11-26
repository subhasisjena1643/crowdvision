"""
Setup script for CrowdVision
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crowdvision",
    version="0.1.0",
    author="CrowdVision Team",
    description="AI/ML-powered situational awareness platform for event safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "mlflow>=2.8.0",
    ],
)
