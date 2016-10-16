
S.lambda <- function(y, lambda){
  # Computes the soft thresholding estimator
  # ----------------------------------------
  # Args: 
  #   - y: vactor of the observations
  #   - lambda: penalization parameter (threshold)
  # Returns: 
  #   - theta: the soft thresholding estimator
  # ------------------------------------------
  theta <- sign(y) * pmax(rep(0, length(y)), abs(y) - lambda)
  return (theta)
}

MSE.comp <- function(y.hat, y){
  # Computes the MSE of predicting y with y.hat
  # -------------------------------------------
  # Args: 
  #   - y.hat: predicted values
  #   - y: true values 
  # Returns: 
  #   - MSE: the MSE
  # ----------------
  return (mean((y - y.hat)^2))
}

kfold.CV <- function(X, y, k, lambda){
  # Computes the k-folds Cross Validation
  # -------------------------------------
  # Args: 
  #   - X: matrix of the features (n*p)
  #   - y: response vector (length n)
  #   - k: number of folds
  #   - lambda: vector of penalization parameters to try
  # Returns: 
  #   - mean.MSE: mean of the estimated test errors for each value of lambda
  #   - sd.MSE: std. dev. of the estimated test errors for each value of lambda
  # ---------------------------------------------------------------------------
  n <- dim(X)[1]
  folds <- cut(1:n, breaks = k, labels=FALSE)
  idx <- sample(folds)
  
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
  # Computes the Cp Mallows
  # -----------------------
  # Args: 
  #   - X: matrix of the features (n*p)
  #   - y: response vector (length n)
  #   - lambda: vector of penalization parameters to try
  # Returns: 
  #   - Cp: Mallow's Cp for each value of lambda
  # --------------------------------------------
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

