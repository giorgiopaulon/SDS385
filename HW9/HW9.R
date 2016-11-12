
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())

source('HW9/rankapprox.R')

##############################
### Test on simulated data ###
##############################

n <- 40
p <- 20

X <- matrix(rnorm(n*p, sd = 3), n, p, byrow = T)

# If we want u and v to be equally sparse
c <- .1
c1 <- c*sqrt(n)
c2 <- c*sqrt(p)

rankapprox <- PMD(X, c1, c2)

rankapprox$u
rankapprox$v

##############################
### Test on marketing data ###
##############################

data <- read.csv(file = '../SDS385-master/data/social_marketing.csv')
data <- as.matrix(data[,-1])

n <- dim(data)[1]
p <- dim(data)[2]

# Anscombe transformation
data <- 2 * sqrt(data + 3/8)

# If we want u and v to be equally sparse
c <- 0.5
c1 <- c*sqrt(n)
c2 <- c*sqrt(p)

rankapprox <- PMD(data, c1, c2)

idx <- sort(rankapprox$v, index.return = T, decreasing = T)$ix
cbind(colnames(data)[idx], rankapprox$v[idx])


