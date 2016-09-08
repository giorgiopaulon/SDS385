
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

### STOCHASTIC GRADIENT DESCENT FOR LOGISTIC REGRESSION

source('../SDS385/HW1/lin_solver.R')
source('../SDS385/HW1/descent_methods.R')
source('../SDS385/HW2/sgd.R')

# (C)

# TRIAL ON SIMULATED DATA

N = 200
P = 10
mi = rep(1,N)

set.seed(123)
X = matrix(rnorm(N*P), nrow = N, ncol = P)
beta = rnorm(P, mean = 0, sd = 1)

y = rbinom(N, 1, comp_wi(X, beta))

maxiter = 100000
tol = 1E-6
alpha = 0.01
beta0 <- rep(0, P)

# Newton method using, as a diagnostic of convergence, the log-likelihood of the 
# entire dataset
newton <- newton_descent(y, X, mi, beta0, maxiter, tol)

# SGD method using a constant step size alpha and, as a diagnostic of convergence, 
# the simple moving average of the individual log-likelihoods
sgd <- SGD(y, X, mi, beta0, maxiter, alpha, tol)

rbind(beta,
newton$beta,
sgd$beta[dim(sgd$beta)[1],])

# SGD method using a Robbins-Monro rule for step sizes and, as a diagnostic of 
# convergence, the simple moving average of the individual log-likelihoods
alpha <- 0.95
C <- 0.5
t0 <- 1
sgd_robbins <- SGD_Robbins(y, X, mi, beta0, maxiter, C, t0, alpha, tol)


plot(newton$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
plot(sgd$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative individual Log-likelihood')
plot(sgd_robbins$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')








# TRIAL ON REAL DATA

wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- cbind(rep(1,N),scale(as.matrix(wdbc[,3:12])))
colnames(X)[1] <- 'Intercept'
P <- dim(X)[2]
mi <- rep(1, N)

glm1 = glm(y~X[,-1], family='binomial')
glm1$coefficients

maxiter = 100000
tol = 1E-10
alpha = 0.01
beta0 <- rep(0, P)

# Newton method using, as a diagnostic of convergence, the log-likelihood of the 
# entire dataset
gradient <- gradient_descent(y, X, mi, beta0, maxiter, alpha, tol)

# Newton method using, as a diagnostic of convergence, the log-likelihood of the 
# entire dataset
newton <- newton_descent(y, X, mi, beta0, maxiter, tol)

# SGD method using a constant step size alpha and, as a diagnostic of convergence, 
# the simple moving average of the individual log-likelihoods
sgd <- SGD(y, X, mi, beta0, maxiter, 1, tol)

par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(sgd$beta[seq(1,maxiter,by=20),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=newton$beta[i],col='red',lwd=2)
}

plot(sgd$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative individual Log-likelihood')

rbind(glm1$coefficients, 
      gradient$beta,
      newton$beta,
      sgd$beta[dim(sgd$beta)[1],])
      #sgd_robbins$beta[dim(sgd_robbins$beta)[1],])


# SGD method using a Robbins-Monro rule for step sizes and, as a diagnostic of 
# convergence, the simple moving average of the individual log-likelihoods
alpha <- 0.99
C <- 8000
t0 <- 2000

plot(1:maxiter,C*(1:maxiter+t0)^(-alpha),type='l',xlab='Iterations',ylab='',main=bquote('Values of'~gamma^(t)),lwd=2)
(C*(1:maxiter+t0)^(-alpha))[1000]
(C*(1:maxiter+t0)^(-alpha))[maxiter]

sgd_robbins <- SGD_Robbins(y, X, mi, beta0, maxiter, C, t0, alpha, tol)

par(mar=c(4,2,2,2),mfrow=c(3,4))
for (i in 1:11){
  plot(sgd_robbins$beta[seq(1,maxiter,by=20),i],type='l',ylab='',xlab='Iterations',main=bquote('Values of'~beta[.(i)]))
  abline(h=newton$beta[i],col='red',lwd=2)
}

plot(sgd_robbins$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative individual Log-likelihood')

rbind(gradient$beta,
  newton$beta,
  sgd$beta[dim(sgd$beta)[1],],
  sgd_robbins$beta[dim(sgd_robbins$beta)[1],])


plot(gradient$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
plot(newton$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')
plot(sgd$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative individual Log-likelihood')
plot(sgd_robbins$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')

