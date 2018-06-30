from setuptools import setup

readme = open('README.md').read()

setup(
    name="tdcca",
    version="0.0.0",
    description="Implementations of Causal Dynamic Network Analysis of fMRI",
    author="Xuefei Cao, Xi Luo, Bjorn Sandstede",
    author_email="xcstf01@gmail.com",
    packages=['tdcca'],
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=[
        "matplotlib>=1.5.3",
        "numpy>=1.11.1",
        "scipy>=0.19.0",
        "six>=1.10.0",
        "scikit-learn>=0.18.1",
        "prox_tv>=3.2.1"
    ],
    url='https://github.com/xuefeicao/tdcca',
    include_package_data=True,
    )
