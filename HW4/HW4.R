
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

library('Rcpp')
library('RcppArmadillo')
library('RcppEigen')
library('Matrix')


###########################
### IMPLEMENTATION IN R ###
###########################

source('HW4/sgd_minibatch_adagrad.R')

# Import data
wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
# Scale data and bind a column of 1 (for the intercept)
X <- cbind(rep(1,N),scale(as.matrix(wdbc[,3:12])))
colnames(X)[1] <- 'Intercept'
P <- dim(X)[2]
mi <- rep(1, N)

glm1 = glm(y~X[,-1], family='binomial')

maxiter = 500000
tol = 1E-20
beta0 <- rep(0, P)

# Stochastic gradient descent + linesearch
sgd <- SGD.linesearch(y, X, mi, beta0, maxiter, tol)

# Plot of the output
par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(sgd$beta[seq(1,dim(sgd$beta)[1], length.out = 5000),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=glm1$coefficients[i],col='red',lwd=2)
}

plot(sgd$ll[seq(1,length(sgd$ll), length.out = 5000)], type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
# This algorithm does not perform well: the estimated values are around the true values but the step size 
# is still too large after a lot of iterations

maxiter = 1000000
# Adaptive gradient descent
adagrad <- AdaGrad(y, X, mi, beta0, maxiter, tol)

# Plot of the output
par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(adagrad$beta[seq(1,dim(adagrad$beta)[1],length.out=5000),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=glm1$coefficients[i],col='red',lwd=2)
}
plot(adagrad$ll[seq(1,length(adagrad$ll), length.out = 5000)], type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
# This algorithm performs much better, but in R it is still too slow to be applied to a big dataset


#############################
### IMPLEMENTATION IN C++ ###
#############################

rm(list=ls())

# We compile the implementation using the library Eigen (there is another implementation using Armadillo, 
# but it is slower).
Rcpp::sourceCpp('./HW4/adagrad_eigen.cc')

# First we try it out on the small dataset
wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- scale(as.matrix(wdbc[,3:12]))
P <- dim(X)[2]
mi <- rep(1, N)
# We exploit the sparse format
X.sp <- Matrix(t(X), sparse = T)

glm1 = glm(y~X, family='binomial')

eta <- 1
lambda <- 0
beta0 <- rep(0, P)

ada <- Ada_Grad(X.sp, y, mi, beta0, eta, npass = 50000, lambda)

c(ada$alpha, ada$beta) # print the estimated coefficients



# Let us try it on the big dataset
rm(list=ls())

Rcpp::sourceCpp('./HW4/adagrad_eigen.cc')

X <- readRDS(file='./HW4/url_tX.rds')
y <- readRDS(file='./HW4/url_y.rds')
N <- length(y)
mi <- rep(1, N)
P <- dim(X)[1]

eta <- 2
lambda <- 1E-6
beta0 <- rep(0, P)

ada <- Ada_Grad(X, y, mi, beta0, eta, npass = 2, lambda)

plot(1:5000, ada$loglik[seq(1, length(ada$loglik), length.out = 5000)], type = 'l', lwd = 2)

# What if we wanted to predict the in-sample error?
esp <- exp(ada$alpha + crossprod(X, ada$beta))
y.hat <- esp/(1 + esp)

y.pred <- array(NA, dim = length(y.hat))
y.pred[which(y.hat > 0.5)] <- 1
y.pred[which(y.hat <= 0.5)] <- 0

prec <- sum(y == y.pred)/length(y)
# The precision of the classifier is around 98%



###############################
### CHOICE OF LAMBDA VIA CV ###
###############################

# rm(list=ls())
# 
# Rcpp::sourceCpp('./HW4/adagrad_iterators_eigen.cc')
# 
# X <- readRDS(file='../SDS385-master/data/url_tX.rds')
# y <- readRDS(file='../SDS385-master/data/url_y.rds')
# N <- length(y)
# mi <- rep(1, N)
# P <- dim(X)[1]
# 
# # Scale so that they have equivalent L2 norm: TO PRESERVE SPARSITY!
# 
# eta <- 1
# lambda.vec <- c(0, 1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3)
# beta0 <- rep(0, P)
# n.folds = 5
# 
# idx <- sample(rep(1:n.folds, length.out = N))
# prec <- matrix(NA, nrow = n.folds, ncol = length(lambda.vec))
# k <- 0
# 
# # (1) Register a parallel
# # Backend doMC
# # registerDoMC(n.cores)
# # (2) Foreach has errors in R gui (run it from the command line!)
# # foreach (j in 1:length(lambda.vec)) %do%
# for (j in 1:length(lambda.vec)){
#   for (i in 1:n.folds){
#     # Select the test set indexes
#     test <- idx == i
#     
#     # Run the algorithm
#     ada <- Ada_Grad(X[,!test ], y[!test], mi[!test], beta0, eta, npass = 1, lambda.vec[j])
#     
#     # Estimate y.hat
#     esp <- exp(ada$alpha + crossprod(X[, test], ada$beta))
#     y.hat <- esp/(1 + esp)
#     
#     # Estimate the y
#     y.pred <- array(NA, dim = length(y.hat))
#     y.pred[which(y.hat > 0.5)] <- 1
#     y.pred[which(y.hat <= 0.5)] <- 0
#     prec[i,j] <- sum(y[test] == y.pred)/length(y.hat)
#     k <- k+1
#     cat("Iteration: ", k)
#   }
# }
# 
# colnames(prec) <- rep("NA", length(lambda.vec))
# for (i in 1:length(lambda.vec))
#   colnames(prec)[i] <- paste("lambda = ", lambda.vec[i], sep = '')


# The code above is commented because it takes a long time to run. I saved the result in the 
# R object I load below.

load(file = './HW4/cv_choice.Rdata')

par(mar=c(4,4,2,2), cex = 1.2)
plotCI(x = 1:length(lambda.vec), y = colMeans(prec), uiw = apply(prec, 2, sd), ylim=c(0.935, 0.99), col = 'orange', pch = 16, scol = 'gray', lwd = 3, xlab = '')
abline(v = which.max(colMeans(prec)), lty = 2)
lines(1:length(lambda.vec), colMeans(prec), lwd = 2, col = 'orange')
