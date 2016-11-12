
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

library(lattice)
library(Matrix)
library(emulator)
library(DRIP)

source('HW8/smoothing.R')

data <- as.matrix(read.csv(file = '../SDS385-master/data/fmri_z.csv'))

n <- dim(data)[1]
p <- dim(data)[2]

# Plot the original data
image(Matrix(t(data)), sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

y <- Matrix(as.vector(t(data)))
D <- makeD2_sparse(n, p)
L <- crossprod(D)

lambda <- 1

x <- laplacian.direct(y, L, lambda)
x1 <- laplacian.jacobi(y, L, lambda, maxiter = 500)
x2 <- laplacian.gaussseidel(y, L, lambda, maxiter = 500)
x3 <- fusedlasso.admm(y, D, rep(0, length(y)), lambda, rho = lambda/2, maxiter = 500)
x4 <- fusedlasso.refined(y, D, rep(0, length(y)), lambda, rho = lambda/2)

# Reshape x in matricial form
x <- Matrix(as.vector(x), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x[t(data) == 0] <- 0
# Plot the smoothed data
image(x, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

# Reshape x in matricial form 
x1 <- Matrix(as.vector(x1), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x1[t(data) == 0] <- 0
# Plot the smoothed data
image(x1, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

# Reshape x in matricial form
x2.pl <- Matrix(as.vector(x2), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x2.pl[t(data) == 0] <- 0
# Plot the smoothed data
image(x2.pl, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

# Reshape x in matricial form
x3.pl <- Matrix(as.vector(x3$x), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x3.pl[t(data) == 0] <- 0
# Plot the smoothed data
image(x3.pl, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

par(mar=c(4,4,2,2))
plot(x3$obj, type = 'l', col = 'blue', lwd = 2, log = 'x', xlab = 'Iterations', ylab = 'Objective')
points(length(x2$obj), x2$obj[length(x2$obj)], pch = 16, cex = 0.8, col = 'blue')



#################################
### CHOICE OF LAMBDA BY LOOCV ###
#################################


lambda.grid <- seq(0, 1, length.out = 50)
LOOCV <- array(NA, dim = length(lambda.grid))
for (i in 1:length(lambda.grid)){
  LOOCV[i] <- laplacian.direct(y, L, lambda.grid[i])$LOOCV
}
plot(lambda.grid, LOOCV, type = 'l', col = 'dodgerblue4', lwd = 2)

x1 <- laplacian.direct(y, L, lambda.grid[which.min(LOOCV)])$x

# Reshape x in matricial form 
x1 <- Matrix(as.vector(x1), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x1[t(data) == 0] <- 0
# Plot the smoothed data
image(x1, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)


data(lena)
data <- lena

n <- dim(data)[1]
p <- dim(data)[2]

# Plot the original data
image(Matrix(t(data)[dim(data)[1]:1,]), sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

y <- Matrix(as.vector(t(data)))
D <- makeD2_sparse(n, p)
L <- crossprod(D)

lambda <- 5

x <- laplacian.direct(y, L, lambda)

# Reshape x in matricial form
x <- Matrix(as.vector(x), nrow = p, ncol = p)
# Plot the smoothed data
image(x[dim(data)[1]:1,], sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)

x2 <- fusedlasso.refined(y, D, rep(0, length(y)), lambda, rho = lambda/2)

# Reshape x in matricial form
x2.pl <- Matrix(as.vector(x2$x), nrow = p, ncol = p)
# Set to 0 the points not detected by the fMRI
x2.pl[t(data) == 0] <- 0
# Plot the smoothed data
image(x2.pl, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)
