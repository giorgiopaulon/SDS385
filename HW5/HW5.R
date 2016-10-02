
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

library(glmnet)
library(plotrix)
source('HW5/cv.R')

### PENALIZED LIKELIHOOD AND SOFT THRESHOLDING

# (A)

# theta.grid <- seq(-5, 5, by = 0.1)

y.grid <- seq(-4, 4, by = 0.01)
lambda <- 2

par(mar=c(4,4,1,1))
plot(y.grid, S.lambda(y.grid, lambda), type = 'l', lwd = 2, col = 'dodgerblue3', xlab = 'y', ylab = bquote(S[lambda]), xlim=range(y.grid), ylim=range(y.grid), asp=1)
plot(y.grid, H.lambda(y.grid, lambda), type = 'l', lwd = 2, col = 'dodgerblue3', xlab = 'y', ylab = bquote(H[lambda]), xlim=range(y.grid), ylim=range(y.grid), asp=1)

# (B)

n <- 100
K <- 0.9

theta <- rnorm(n, 0, 5)
mask <- rbinom(n, 1, K)
theta <- theta * mask
sigma2 <- rgamma(n, shape = 0.5, scale = 1)

y <- rnorm(n, theta, sigma2)

lambda <- 0
theta.hat <- S.lambda(y, sigma2*lambda)

par(mar=c(4,4.5,2,2), cex = 1.5)
plot(theta, theta.hat, xlab = bquote(theta), ylab=bquote(hat(theta)), pch = 1, lwd = 2)
abline(a = 0, b = 1, lwd = 2, col='red')
abline(h=0)

lambda.grid <- seq(0, 5, by = 0.05)
MSE <- array(NA, dim = length(lambda.grid))
for (i in 1:length(lambda.grid)){
  theta.hat <- S.lambda(y, sigma2*lambda.grid[i])
  MSE[i] <- mean((theta.hat - theta)^2)
}

par(mar=c(4,4.5,2,2), cex = 1.2)
plot(lambda.grid, MSE, type = 'l', lwd = 2, col='red', xlab = bquote(lambda))
abline(v=lambda.grid[which.min(MSE)], lwd = 2)

### THE LASSO

# Load and standardize the data
y <- as.numeric(read.csv(file = '../SDS385-master/data/diabetesY.csv', header = F)[,1])
X <- as.matrix(read.csv(file = '../SDS385-master/data/diabetesX.csv', header = T))
X <- scale(X)
y <- scale(y)
n <- dim(X)[1]
p <- dim(X)[2]

# Fit a Lasso regression for a grid of penalty coefficient lambda
fit <- glmnet(X, y, family="gaussian", alpha = 1, nlambda = 100, standardize = FALSE, intercept=FALSE)

# (A)
# Plot the solution paths for each beta_i
par(mar=c(4,4,2,2))
plot(fit, xvar = 'lambda', xlab = bquote(log(lambda)), lwd=2, label = T)

# Compute and track the in sample MSE
y.pred <- predict(fit, newx = X)
MSE <- array(NA, dim = dim(y.pred)[2])
for (i in 1:dim(y.pred)[2]){
  MSE[i] <- MSE.comp(y.pred[,i], y)
}
plot(log(fit$lambda), MSE, type = 'b', pch = 16, xlab = bquote(log(lambda)))
# The minimum of the in sample MSE is reached for the model with all the covariates (lambda approximately
# equal to 0). This is not surprising, as this model is the best in explaining the effect of the 
# covariates on the TRAINING set (but it may be overfitting). 

# (B)
# Choose the number of folds
k <- 10
MOOSE <- kfold.CV(X, y, k, fit$lambda)

# Plot the MOOSE along with the error bars.
plotCI(x = log(fit$lambda), y = MOOSE$mean.MSE, uiw = MOOSE$sd.MSE, col = 'red', pch = 16, scol = 'gray', cex=0.8)
# Trace the line corresponding to the optimal lambda
abline(v = log(fit$lambda[which.min(MOOSE$mean.MSE)]), lwd = 1, col = 1, lty = 2)

# abline(h = min(MOOSE$mean.MSE) + MOOSE$sd.MSE[which.min(MOOSE$mean.MSE)])
# Look for the indexes after the optimal one which have a mean which is still lower than the optimal
# MSE + one dev. std
idx <- which(min(MOOSE$mean.MSE) + MOOSE$sd.MSE[which.min(MOOSE$mean.MSE)] >= MOOSE$mean.MSE &
  1:length(fit$lambda) <= which.min(MOOSE$mean.MSE))
# Pick the first of these indexes which is TRUE
idx <- idx[1]
# Trace the line corresponding to the most conservative model among the ones with optimal lambda 
# 1-std. dev criterion.
abline(v = log(fit$lambda[idx]), lwd = 1, col = 1, lty = 3)


# Just for the sake of completeness, we compare the result with R's built-in function. The two functions
# should give similar results.
cv <- cv.glmnet(X, y, lambda = fit$lambda, nfolds = k)
plot(cv)


# (C)
# We compute and plot the Mallow's Cp in order to pick the best model complexity.
Cp <- Cp.mallows(X, y, fit$lambda)
plot(log(fit$lambda), Cp, type = 'b', pch = 16)

# Comparison between the three methods
par(mar=c(4,2,2,2), cex = 1.4)
plot(log(fit$lambda), MSE, type = 'b', pch = 16, xlab = bquote(log(lambda)), cex = 0.6, col = 'dodgerblue3')
plotCI(x = log(fit$lambda), y = MOOSE$mean.MSE, uiw = MOOSE$sd.MSE, col = 'red', pch = 16, scol = 'gray', add = T, cex = 0.6)
points(log(fit$lambda), Cp, type = 'b', pch = 16, cex = 0.6, col = 'goldenrod2')
legend('topleft', legend = c('MSE', 'MOOSE', 'Cp'), col = c('dodgerblue3', 'red', 'goldenrod2'), pch = 16, cex = 0.8)
abline(v = log(fit$lambda[which.min(MOOSE$mean.MSE)]), lwd = 2, col = 'red', lty = 2)
abline(v = log(fit$lambda[which.min(Cp)]), lwd = 2, col = 'goldenrod2', lty = 2)

# Number of selected covariates in the two cases
fit$df[which.min(MOOSE$mean.MSE)]
fit$df[which.min(Cp)]
fit$df[idx]


