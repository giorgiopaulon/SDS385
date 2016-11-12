setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

source('HW6/proximal_gradient.R')
source('HW7/ADMM.R')
library(glmnet)

### ADAPTIVE DIRECTION METHOD OF MULTIPLIERS
# (B)

# Load and standardize the data
y <- as.numeric(read.csv(file = '../SDS385-master/data/diabetesY.csv', header = F)[,1])
X <- as.matrix(read.csv(file = '../SDS385-master/data/diabetesX.csv', header = T))
X <- scale(X)
y <- scale(y)
n <- dim(X)[1]
p <- dim(X)[2]

beta0 <- rep(0, p)
lambda = 1E-2

lasso1 <- proximal.gradient(y, X, beta0, lambda = lambda, gamma = 1E-4, tol = 1E-12)

lasso2 <- acc.proximal.gradient(y, X, beta0, lambda = lambda, gamma = 1E-4, tol = 1E-12)

lasso3 <- admm(y, X, beta0, lambda, rho = 1000)

fit <- glmnet(X, y, family = 'gaussian', alpha = 1, lambda = lambda, standardize = FALSE, intercept = FALSE)


par(mar=c(4,4,2,2))
plot(lasso1$ll, type = 'l', col = 'blue', lwd = 2, log = 'x', xlab = 'Iterations', ylab = 'Objective')
points(length(lasso1$ll), lasso1$ll[length(lasso1$ll)], pch = 16, cex = 0.8, col = 'blue')
lines(lasso2$ll, col = 'red', type = 'l', lwd = 2)
points(length(lasso2$ll), lasso2$ll[length(lasso2$ll)], pch = 16, cex = 0.8, col = 'red')
lines(lasso3$ll, col = 'green', type = 'l', lwd = 2)
points(length(lasso3$ll), lasso3$ll[length(lasso3$ll)], pch = 16, cex = 0.8, col = 'green')

results <- cbind(lasso1$betas, lasso2$betas, lasso3$x, fit$beta)
colnames(results) <- c('Prox Grad', 'Acc Prox Grad', 'ADMM', 'Glmnet')
results

lambda.grid <- exp(seq(-10, 6, by = 0.5))
betas <- array(NA, dim = c(length(lambda.grid), p))
for (i in 1:length(lambda.grid)){
  lasso <- admm(y, X, beta0, lambda.grid[i], rho = 1000)
  betas[i, ] <- lasso$x
}
fit <- glmnet(X, y, family = 'gaussian', alpha = 1, lambda = lambda.grid[length(lambda.grid):1], standardize = FALSE, intercept = FALSE)

par(mar=c(4,4,2,2))
plot(fit, xvar = 'lambda')
matplot(log(lambda.grid), betas, type = 'l', lty = 1)

