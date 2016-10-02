
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

library('Rcpp')
library('RcppArmadillo')
# library('RcppEigen')
library('Matrix')


###########################
### IMPLEMENTATION IN R ###
###########################

source('HW4/sgd_minibatch_adagrad.R')

wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- cbind(rep(1,N),scale(as.matrix(wdbc[,3:12])))
colnames(X)[1] <- 'Intercept'
P <- dim(X)[2]
mi <- rep(1, N)

glm1 = glm(y~X[,-1], family='binomial')

eta <- 2
lambda <- 0
maxiter = 2000000
tol = 1E-20
beta0 <- rep(0, P)

# Stochastic gradient descent + linesearch
sgd <- SGD.linesearch(y, X, mi, beta0, maxiter, tol)


par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(sgd$beta[seq(1,dim(sgd$beta)[1], by=100),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=glm1$coefficients[i],col='red',lwd=2)
}

plot(sgd$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')


# Adaptive gradient descent
adagrad <- AdaGrad(y, X, mi, beta0, maxiter, tol)


par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(adagrad$beta[seq(1,dim(adagrad$beta)[1],length.out=5000),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=glm1$coefficients[i],col='red',lwd=2)
}

plot(adagrad$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')


#############################
### IMPLEMENTATION IN C++ ###
#############################

rm(list=ls())

Rcpp::sourceCpp('./HW4/adagrad_iterators_eigen.cc')

wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- scale(as.matrix(wdbc[,3:12]))
P <- dim(X)[2]
mi <- rep(1, N)
X.sp <- Matrix(t(X), sparse = T)

glm1 = glm(y~X, family='binomial')

eta <- 1
lambda <- 0
beta0 <- rep(0, P)

ada <- Ada_Grad(X.sp, y, mi, beta0, eta, npass = 10000, lambda)

plot(1:5000, ada$nll_tracker[seq(1, length(ada$nll_tracker), length.out = 5000)], type = 'l', lwd = 2)



rm(list=ls())

Rcpp::sourceCpp('./HW4/adagrad_iterators_eigen.cc')

X <- readRDS(file='../SDS385-master/data/url_tX.rds')
y <- readRDS(file='../SDS385-master/data/url_y.rds')
N <- length(y)
mi <- rep(1, N)
P <- dim(X)[1]

eta <- 2
lambda <- 0
beta0 <- rep(0, P)

ada <- Ada_Grad(X, y, mi, beta0, eta, npass = 1, lambda)

plot(1:5000, ada$nll_tracker[seq(1, length(ada$nll_tracker), length.out = 5000)], type = 'l', lwd = 2)

esp <- exp(ada$alpha + crossprod(X, ada$beta))
y.hat <- esp/(1 + esp)

y.pred <- array(NA, dim = length(y.hat))
y.pred[which(y.hat > 0.5)] <- 1
y.pred[which(y.hat <= 0.5)] <- 0

prec <- sum(y == y.pred)/length(y)


mat <- rbind(colMeans(prec), apply(prec, 2, sd))
rownames(mat) <- c('Mean CV error', 'Std. dev. CV error')

mat <- xtable(mat, digits = 4)


###############################
### CHOICE OF LAMBDA VIA CV ###
###############################

rm(list=ls())

Rcpp::sourceCpp('./HW4/adagrad_iterators_eigen.cc')

X <- readRDS(file='../SDS385-master/data/url_tX.rds')
y <- readRDS(file='../SDS385-master/data/url_y.rds')
N <- length(y)
mi <- rep(1, N)
P <- dim(X)[1]

eta <- 1
lambda.vec <- c(0, 1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3)
beta0 <- rep(0, P)
n.folds = 5

idx <- sample(rep(1:n.folds, length.out = N))
prec <- matrix(NA, nrow = n.folds, ncol = length(lambda.vec))
k <- 0
for (j in 1:length(lambda.vec)){
  for (i in 1:n.folds){
    test <- idx == i
    ada <- Ada_Grad(X[,!test ], y[!test], mi[!test], beta0, eta, npass = 1, lambda.vec[j])
    esp <- exp(ada$alpha + crossprod(X[, test], ada$beta))
    y.hat <- esp/(1 + esp)
    y.pred <- array(NA, dim = length(y.hat))
    y.pred[which(y.hat > 0.5)] <- 1
    y.pred[which(y.hat <= 0.5)] <- 0
    prec[i,j] <- sum(y[test] == y.pred)/length(y.hat)
    k <- k+1
    cat("Iteration: ", k)
  }
}

colnames(prec) <- rep("NA", length(lambda.vec))
for (i in 1:length(lambda.vec))
  colnames(prec)[i] <- paste("lambda = ", lambda.vec[i], sep = '')
save(prec, file = 'cv_choice.Rdata')




par(mar=c(4,4,2,2), cex = 1.2)
plotCI(x = 1:length(lambda.vec), y = colMeans(prec), uiw = apply(prec, 2, sd), ylim=c(0.935, 0.99), col = 'orange', pch = 16, scol = 'gray', lwd = 3, xlab = '')
abline(v = which.max(colMeans(prec)), lty = 2)
lines(1:length(lambda.vec), colMeans(prec), lwd = 2, col = 'orange')
