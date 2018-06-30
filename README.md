# Time-dependent Canonical Correlation Analysis 
Canonical Correlation Analysis is a technique in multivariate data analysis for finding correlation and pairs of vectors that maximizes the correlation between a set of paired variables. Many important problems involve recoding time-dependent observations. In order to understand the coupling dynamics between the two sources, spot trends, detect anomalies, in this paper, we introduce the time-dependent canonical correlation analysis (TDCCA), a method of inferring time-dependent canonical vectors of paired variables.  


## Getting Started
We provide both simulation examples used in our paper. The main computation algorithm is not added into class method for the convience of algorithm development for multi-view data. 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

This package also provides other algorithms to optimize the function, including cvxpy naive optimization, cvxpy naive admm optimization and two other admm algorithms. For details, see the admm_computation.py file. However, these functions have not been tested thoroughly. 

This package is also published in pypi. For a quick installation, try

```
pip install tdcca 
```

### Prerequisites

What things you need to install the software and how to install them

```
See setup.py for details of packages requirements. 
```

### Installing from GitHub


Download the packages by using git clone https://github.com/xuefeicao/tdcca.git

```
python setup.py install
```

If you experience problems related to installing the dependency Matplotlib on OSX, please see https://matplotlib.org/faq/osx_framework.html 

### Intro to our package
After installing our package locally, try to import cdn in your python environment and learn about package's function. 
Note: our package name in pypi is tdcca.
```

```


### Examples
```
The examples subfolder includes simulations provided in the paper. 
```

## Running the tests

The test is going to be added in the future.

## Built With

* Python 2.7

## Compatibility
* python 2.7


## Authors

* **Xuefei Cao** - *Maintainer* - (https://github.com/xuefeicao)
* **Xi Luo** (http://bigcomplexdata.com/)
* **Björn Sandstede** (http://www.dam.brown.edu/people/sandsted/)


## License

This project is licensed under the MIT License - see the LICENSE file for details

