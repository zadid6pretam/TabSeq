from setuptools import setup, find_packages

setup(
    name="tabseq",
    version="0.1",
    author="Zadid Habib",
    author_email="ah00069@mix.wvu.edu",
    description="TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zadid6pretam/TabSeq",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "tensorflow==2.13.0",
        "networkx==3.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
