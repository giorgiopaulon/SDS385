
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

library(microbenchmark)
library(Matrix)

### LINEAR REGRESSION 

# (C)

source('../SDS385/HW1/lin_solver.R')

NPmatrix <- matrix(c(100,10,500,50,3000,500,3000,1000), nrow=4, ncol = 2, byrow=T)
comp_times <- matrix(NA, nrow = 4, ncol = 3)

for (i in 1:4){

  N = NPmatrix[i,1]
  P = NPmatrix[i,2]
  
  set.seed(123)
  X = matrix(rnorm(N*P), nrow = N, ncol = P)
  beta = rnorm(P, mean = 0, sd = 5)
  sigma = 1
  
  y = X%*%beta + rnorm(N, mean = 0, sd = sqrt(sigma))
  
  # Regression estimation
  #W = diag(runif(N, 0, 1))
  W = diag(rep(1, N))
  beta_hat1 <- my_inv(X, y, W)
  beta_hat2 <- my_chol(X, y, W)
  beta_hat3 <- my_QR(X, y, W)
  
  
  mb <- microbenchmark(my_inv(X, y, W), my_chol(X, y, W), my_QR(X, y, W), unit='ms', times = 5L)
  
  comp_times[i,1] <- median(mb$time[mb$expr=='my_inv(X, y, W)'])/10^6
  comp_times[i,2] <- median(mb$time[mb$expr=='my_chol(X, y, W)'])/10^6
  comp_times[i,3] <- median(mb$time[mb$expr=='my_QR(X, y, W)'])/10^6

}

rownames(comp_times) <- c('N = 100; P = 10','N = 500; P = 50','N = 3000; P = 500','N = 3000; P = 1000')
colnames(comp_times) <- c('Inversion','Cholesky','QR')
matplot(comp_times,type='l',lwd=2,xaxt='n',ylab='Computational Time [ms]')
axis(1, at = 1:4, labels = rownames(comp_times), cex.axis = 0.8)
legend('topleft', legend=colnames(comp_times), col=c(1,2,3), lty=1:3, lwd = 2)

# (D)

comp_times2 <- matrix(NA, nrow = 4, ncol = 3)

K_vec <- c(0.01,0.1,0.5,1)
for (i in 1:4){

  N = 1000
  P = 500
  
  set.seed(123)
  X = matrix(rnorm(N*P), nrow = N, ncol = P)
  mask = matrix(rbinom(N*P, 1, K_vec[i]), nrow = N, ncol = P)
  X = mask*X
  beta = rnorm(P, mean = 0, sd = 5)
  sigma = 1
  
  y = X%*%beta + rnorm(N, mean = 0, sd = sqrt(sigma))
  
  # Regression estimation
  W = diag(rep(1, N))
  beta_hat1 <- my_inv(X, y, W)
  beta_hat2 <- my_chol(X, y, W)
  beta_hat3 <- my_inv_sparse(X, y, W)

  mb <- microbenchmark(my_inv(X, y, W), my_chol(X, y, W), my_inv_sparse(X, y, W), unit='ms', times = 5L)
  
  comp_times2[i,1] <- median(mb$time[mb$expr=='my_inv(X, y, W)'])/10^6
  comp_times2[i,2] <- median(mb$time[mb$expr=='my_chol(X, y, W)'])/10^6
  comp_times2[i,3] <- median(mb$time[mb$expr=='my_inv_sparse(X, y, W)'])/10^6
  
}

rownames(comp_times2) <- c('K = 0.01','K = 0.1', 'K = 0.5', 'K = 1')
colnames(comp_times2) <- c('Inversion','Cholesky','Sparse Inversion')
matplot(comp_times2,type='l',lwd=2,xaxt='n',ylab='Computational Time [ms]')
axis(1, at = 1:4, labels = rownames(comp_times2), cex.axis = 1)
legend('topleft', legend=colnames(comp_times2), col=c(1,2,3), lty=1:3, lwd = 2)

### GENERALISED LINEAR MODELS 

# (B)

source('../SDS385/HW1/descent_methods.R')


# # TRIAL ON SIMULATED DATA
# 
# N = 200
# P = 10
# mi = rep(1,N)
# 
# set.seed(123)
# X = matrix(rnorm(N*P), nrow = N, ncol = P)
# beta = rnorm(P, mean = 0, sd = 1)
# 
# y = rbinom(N, 1, comp_wi(X, beta))
# 
# maxiter = 5000
# alpha = 0.01
# tol = 10E-3
# beta0 <- rep(-10, P)
# 
# gradient <- gradient_descent(y, X, mi, beta0, maxiter, alpha, tol)
# 
# plot(gradient$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')

# TRIAL ON REAL DATA

wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- cbind(rep(1,N),scale(as.matrix(wdbc[,3:12])))
colnames(X)[1] <- 'Intercept'
P <- dim(X)[2]
mi <- rep(1, N)

maxiter = 100000
tol = 10E-5
alpha = 0.01
beta0 <- rep(0, P)

gradient <- gradient_descent(y, X, mi, beta0, maxiter, alpha, tol)
newton <- newton_descent(y, X, mi, beta0, maxiter, tol)

par(mar=c(4,4,2,2))
plot(gradient$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
plot(newton$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')

