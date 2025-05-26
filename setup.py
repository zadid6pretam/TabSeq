from setuptools import setup, find_packages

setup(
    name="TabSeq",
    version="0.1.5",
    author="Zadid Habib",
    author_email="ah00069@mix.wvu.edu",
    description="TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zadid6pretam/TabSeq",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22,<1.25",
        "pandas>=2.0.0,<2.2",
        "scikit-learn>=1.2,<1.4",
        "tensorflow>=2.12,<2.14",
        "networkx>=3.0,<3.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
