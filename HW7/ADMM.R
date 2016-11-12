

log.lik <- function(beta, y, X, lambda){
  # Computes the objective function of the logit model
  # --------------------------------------------------
  # Args: 
  #   - beta: regression parameters (length p)
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - lambda: penalization parameter
  # Returns: 
  #   - ll: the scalar associated with the objective function at beta
  # -----------------------------------------------------------------
  Xbeta <- X %*% beta
  ll <- (1/2)*crossprod(y - Xbeta) + lambda * sum(abs(beta))
  return(ll)
}


soft.thresholding <- function(x, lambda){
  # Computes the soft thresholding estimator
  # ----------------------------------------
  # Args: 
  #   - x: vector of the observations
  #   - lambda: penalization parameter (threshold)
  # Returns: 
  #   - theta: the soft thresholding estimator
  # ------------------------------------------
  theta <- sign(x) * pmax(rep(0, length(x)), abs(x) - lambda)
  return (theta)
}


admm <- function(b, A, x0, lambda, rho = 1, maxiter = 5000, tol.abs = 1E-10, tol.rel = 1E-10){
  # ADMM that allows to compute the Lasso regression estimates
  # ----------------------------------------------------------
  # Args: 
  #   - b: response vector (length n)
  #   - A: matrix of the features (n*p)
  #   - x0: initial values of beta
  #   - lambda: penalization parameter for the lasso (threshold)
  #   - rho: step size
  #   - maxiter: maximum number of iterations
  #   - tol: tolerance defining convergence
  # Returns: 
  #   - ll: values of the objective function for each iteration
  #   - betas: values of the beta parameters for each iteration
  # -----------------------------------------------------------
  
  # Check if the inputs have the correct sizes
  if ((length(b) != dim(A)[1]) || (length(x0) != dim(A)[2])){
    stop("Incorrect input dimensions.")
  }
  
  # Rescaling the threshold to make it comparable to glmnet
  lambda <- lambda * length(b)
  p <- length(x0)
  
  # Initialize the data structures
  x <- array(NA, dim = p)
  x <- x0 # Initial guess
  z <- array(NA, dim = p)
  z <- x0
  u <- array(NA, dim = p)
  u <- rep(0, p)
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(x, b, A, lambda)
  
  # Precaching operations
  R = chol(crossprod(A) + diag(rep(rho, p)))
  Atb <- crossprod(A, b)
  
  for (iter in 2:maxiter){
    
    zold <- z
    
    # x - minimization step
    a = forwardsolve(t(R), Atb + rho*(z - u))
    x = backsolve(R, a)
    
    # z - minimization step
    z <- soft.thresholding(x + u, lambda/rho)
    
    # u - minimization step
    r <- x - z
    u <- u + r

    # Compute log-likelihood
    ll[iter] <- log.lik(x, b, A, lambda)
    
    rnorm <- sqrt(sum(r^2))
    snorm <- rho * sqrt(sum((z - zold)^2))
    eprim <- sqrt(p) * tol.abs + tol.rel * max(c(sqrt(sum(x^2)), sqrt(sum(z^2)), 0))
    edual <- sqrt(p) * tol.abs + tol.rel * rho * sqrt(sum(u^2))
    # Convergence check
    if ( (rnorm < eprim) & (snorm < edual) ){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      break;
    }
    else if ( (iter == maxiter) & ((rnorm >= eprim) || (snorm >= edual)) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("ll" = ll, "x" = z))
}

