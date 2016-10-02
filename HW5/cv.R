
log.lik <- function(y, lambda, theta){
  obj <- (1/2) * (y - theta)^2 + lambda * abs(theta)
  return (obj)
}

S.lambda <- function(y, lambda){
  theta <- sign(y) * pmax(rep(0, length(y)), abs(y) - lambda)
  return (theta)
}

H.lambda <- function(y, lambda){
  theta <- array(NA, dim = length(y))
  for (i in 1:length(y)){
    if (y[i] >= lambda){
      theta[i] <- y[i]
    }
    else{
      theta[i] <- 0
    }
  }
  return (theta)
}



MSE.comp <- function(y.hat, y){
  return (mean((y - y.hat)^2))
}

kfold.CV <- function(X, y, k, lambda){
  n <- dim(X)[1]
  idx <- sample(rep(1:k, length.out = n))
  
  MSE <- array(NA, dim = c(k, length(lambda)))
  for (i in 1:k){
    test <- idx == i
    X.tr <- X[!test, ]
    y.tr <- y[!test]
    fit <- glmnet(X.tr, y.tr, family="gaussian", alpha = 1, lambda = lambda, intercept=FALSE)
    X.test <- X[test, ]
    y.test <- y[test]
    if (sum(test) == 1){
      y.pred <- predict(fit, newx = t(data.matrix(X.test)))
    }
    else{
      y.pred <- predict(fit, newx = X.test)
    }
    MSE[i,] <- apply(y.pred, 2, MSE.comp, y.test)
  }
  return (list("mean.MSE" = colMeans(MSE), "sd.MSE" = apply(MSE, 2, sd)/sqrt(k)))
}

Cp.mallows <- function(X, y, lambda){
  n <- length(y)
  p <- dim(X)[2]
  fit <- glmnet(X, y, family="gaussian", alpha = 1, lambda = 0, standardize = FALSE, intercept = FALSE)
  sigma2 <- sum((y - predict(fit, newx = X))^2)/(n - p)
  
  fit <- glmnet(X, y, family="gaussian", alpha = 1, lambda = lambda, standardize = FALSE, intercept=FALSE)
  y.pred <- predict(fit, newx = X)
  MSE <- array(NA, dim = length(lambda))
  for (i in 1:length(lambda)){
    MSE[i] <- MSE.comp(y.pred[,i], y)
  }
  Cp <- MSE + 2*(fit$df/n)*sigma2
  return (Cp)
}

