
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

lasso <- proximal.gradient(y, X, beta0, lambda)
  

