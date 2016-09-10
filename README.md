# SDS 385, Fall 2016
I am Giorgio Paulon, 1st year PhD student in Statistics at UT Austin. This is my personal page for the Big Data class with Prof. James Scott.

## Exercise 1
In the [first homework](https://github.com/gpaulon/SDS385/tree/master/HW1), we firstly adressed the problem of linear regression in the context of weighted least squares. We tested three different matrix factorization that can deal with large datasets, and afterwards we implemented methods for dealing with large sparse matrices. A benchmarking of the different procedures has been carried out. 

Secondly, the problem of logistic regression has been tackled. Two approaches minimizing the negative log-likelihood of the data have been performed: the gradient descent and the Newton-Raphson method.

## Exercise 2
In the [second homework](https://github.com/gpaulon/SDS385/tree/master/HW2), we implement Stochastic gradient descent (SGD), an algorithm that allows to approximate the computation of the gradient with a faster routine, using an unbiased estimate of the gradient. 

The SGD for logistic regression has been succesfully implemented. This algorithm, despite being faster than gradient descent and Newton method, has crucial issues, such as the choice of the step size. In a more sophisticated version, the step size can vary along the number of iterations. This is a very important step in order to ensure convergence. 
