
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())


source('HW6/proximal_gradient.R')

### THE PROXIMAL GRADIENT METHOD 
# (B)

# Load and standardize the data
y <- as.numeric(read.csv(file = '../SDS385-master/data/diabetesY.csv', header = F)[,1])
X <- as.matrix(read.csv(file = '../SDS385-master/data/diabetesX.csv', header = T))
X <- scale(X)
y <- scale(y)
n <- dim(X)[1]
p <- dim(X)[2]

beta0 <- rep(0, p)
lambda = 0.5

lasso1 <- proximal.gradient(y, X, beta0, lambda = 1, gamma = 1E-2, maxiter = 10000, tol = 1E-8)

par(mar=c(4,4,2,2))
plot(lasso1$ll, type = 'l', col = 'blue', lwd = 2, log = 'x', xlab = 'Iterations', ylab = 'Objective')
points(length(lasso1$ll), lasso1$ll[length(lasso1$ll)], pch = 16, cex = 0.8, col = 'blue')

lasso2 <- acc.proximal.gradient(y, X, beta0, lambda = 1, gamma = 1E-4, maxiter = 10000, tol = 1E-8)

lines(lasso2$ll, col = 'red', type = 'l', lwd = 2)
points(length(lasso2$ll), lasso2$ll[length(lasso2$ll)], pch = 16, cex = 0.8, col = 'red')

fit <- glmnet(X, y, family = 'gaussian', alpha = 1, lambda = 1/n, standardize = FALSE, intercept = FALSE)

cbind(lasso1$betas, lasso2$betas, fit$beta)
