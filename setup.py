from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="digantara-instruments",
    version="0.1.0",
    author="Anirudh Iyengar",
    author_email="anirudh.iyengar@digantara.co.in",
    description="Instrument control and automation tools for Digantara's testing infrastructure",
    maintainer="Digantara Research and Technologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/digantara/instruments",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy>=2.0.0,<3.0.0',
        'pandas>=2.2.0,<3.0.0',
        'matplotlib>=3.8.0,<4.0.0',
        'Pillow>=10.0.0,<11.0.0',
        'pyvisa>=1.13.0,<2.0.0',
        'pyvisa-py>=0.7.2,<1.0.0',
        'pyserial>=3.5,<4.0.0',
        'pyusb>=1.2.1,<2.0.0',
        'gradio>=4.0.0,<5.0.0',
        'huggingface_hub>=0.25.2,<1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.0.0',
            'mypy>=1.0.0',
            'sphinx>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'dig-instruments=instrument_control.cli:main',
        ],
    },
)
