
setwd('~/Desktop/Semester 1/Stat Models for Big Data/SDS385/')
rm(list=ls())


### LINE SEARCH

# (B)

source('../SDS385/HW3/line_search.R')

# TRIAL ON REAL DATA

wdbc <- read.csv(file = '../SDS385-master/data/wdbc.csv', header = F)

y <- as.numeric(wdbc[,2])-1 # 0 for benign, 1 for malign
N <- length(y)
X <- cbind(rep(1,N),scale(as.matrix(wdbc[,3:12])))
colnames(X)[1] <- 'Intercept'
P <- dim(X)[2]
mi <- rep(1, N)

glm1 = glm(y~X[,-1], family='binomial')


maxiter = 10000
tol = 10E-10
beta0 <- rep(0, P)

gradient <- GD.line.search(y, X, mi, beta0, maxiter, tol)

newton <- quasi.newton(y, X, mi, beta0, maxiter, tol)

par(mar=c(4,4,3,2))
barplot(table(gradient$alpha), main='Barplot of the chosen optimal step sizes')
barplot(table(newton$alpha), main='Barplot of the chosen optimal step sizes')

par(mar=c(4,4,3,2))
plot(gradient$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood', log='x')
plot(newton$ll, type = 'l', lwd = 2, col = 'red', xlab = 'Iterations', ylab = 'Negative Log-likelihood')

results <- rbind(glm1$coefficients, gradient$beta, newton$beta)
for (i in 0:(dim(results)[2]-1))
  colnames(results)[i+1] <- paste('Beta', i, sep='')
rownames(results) <- c('Glm','Gradient','Newton')
results
