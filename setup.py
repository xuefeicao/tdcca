from setuptools import setup

readme = open('README.md').read()

setup(
    name="tdcca",
    version="0.0.0",
    description="Time-dependent Canonical Correlation Analysis",
    author="Xuefei Cao, Jun Ke, Xi Luo, Bjorn Sandstede",
    packages=['tdcca'],
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=[
        "matplotlib>=1.5.3",
        "numpy>=1.11.1",
        "scipy>=0.19.0",
        "six>=1.10.0",
        "scikit-learn>=0.18.1",
        "prox_tv>=3.2.1",
        "pathos>=0.2.2", 

    ],
    url='https://github.com/xuefeicao/tdcca',
    include_package_data=True,
    )
