"""
INTEGRA — Automated Lecture Video Summarization
A research project for condensing 60-minute IT theory lectures into 10-minute summaries.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lecture-video-summarizer",
    version="0.1.0",
    author="Integra",
    author_email="integra@uom.lk",
    description="Automated lecture video summarization using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rashmiJayawardhana/lecture-video-summarizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lecture-summarizer=pipeline.summarizer:main",
        ],
    },
)
