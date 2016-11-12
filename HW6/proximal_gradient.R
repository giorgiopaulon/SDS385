
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


grad.loglik <- function(beta, y, X){
  # Computes the gradient of negative log-likelihood of the logit model
  # -------------------------------------------------------------------
  # Args: 
  #   - beta: regression parameters (length p)
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  # Returns: 
  #   - grad.ll: the gradient vector of the negative log-lik of the data (length p)
  # -------------------------------------------------------------------------------
  grad.ll <- array(NA, dim = length(beta))
  Xbeta <- X %*% beta
  grad.ll <- - crossprod(X, as.numeric(y - Xbeta))
  return(grad.ll)
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


proximal.gradient <- function(y, X, beta0, lambda, gamma = 1E-4, maxiter = 50000, tol = 1E-10){
  # Proximal gradient algorithm that allows to compute the Lasso regression estimates
  # ---------------------------------------------------------------------------------
  # Args: 
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - beta0: initial values of beta
  #   - lambda: penalization parameter for the lasso (threshold)
  #   - gamma: proximal operator parameter
  #   - maxiter: maximum number of iterations
  #   - tol: tolerance defining convergence
  # Returns: 
  #   - ll: values of the negative log-likelihood for each iteration
  #   - betas: values of the beta parameters for each iteration
  # -----------------------------------------------------------
  
  # Check if the inputs have the correct sizes
  if ((length(y) != dim(X)[1]) || (length(beta0) != dim(X)[2])){
    stop("Incorrect input dimensions.")
  }
  
  lambda <- lambda * length(y)
  
  # Initialize the data structures
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1,], y, X, lambda)
  
  for (iter in 2:maxiter){
    
    # Compute the gradient
    gradient <- grad.loglik(betas[iter-1, ], y, X)
    u <- betas[iter-1, ] - gamma * gradient
    
    # Update beta
    betas[iter, ] <- soft.thresholding(u, gamma * lambda)
    
    # Compute log-likelihood
    ll[iter] <- log.lik(betas[iter,], y, X, lambda)
    
    # Convergence check
    if ( abs(ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-3) < tol ){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }
    else if ( (iter == maxiter) & (abs(ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-3) >= tol) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("ll" = ll, "betas" = betas[iter,]))
}


acc.proximal.gradient <- function(y, X, beta0, lambda, gamma = 1E-4, maxiter = 50000, tol = 1E-10){
  # Accelerated proximal gradient algorithm that allows to compute the Lasso regression estimates
  # ---------------------------------------------------------------------------------------------
  # Args: 
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - beta0: initial values of beta
  #   - lambda: penalization parameter for the lasso (threshold)
  #   - gamma: proximal operator parameter
  #   - maxiter: maximum number of iterations
  #   - tol: tolerance defining convergence
  # Returns: 
  #   - ll: values of the objective function for each iteration
  #   - betas: values of the beta parameters for each iteration
  # -----------------------------------------------------------
  
  # Check if the inputs have the correct sizes
  if ((length(y) != dim(X)[1]) || (length(beta0) != dim(X)[2])){
    stop("Incorrect input dimensions.")
  }
  
  lambda <- lambda * length(y)
  
  # We could precache the operations XtX, Xtbeta in order to compute the gradient
  
  # Initialize the data structures
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  z <- array(NA, dim = length(beta0))
  z <- beta0
  s <- array(NA, dim = maxiter)
  s[1] <- 1
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1,], y, X, lambda)
  
  for (iter in 2:maxiter){
    
    # Compute the gradient
    gradient <- grad.loglik(z, y, X)
    u <- z - gamma * gradient
    
    # Update beta
    betas[iter, ] <- soft.thresholding(u, gamma * lambda)
    s[iter] <- (1 + sqrt(1 + 4 * s[iter-1]^2))/2
    z <- betas[iter, ] + ((s[iter-1] - 1)/s[iter]) * (betas[iter, ] - betas[iter-1, ])
      
    # Compute log-likelihood
    ll[iter] <- log.lik(betas[iter,], y, X, lambda)
    
    # Convergence check
    if ( abs(ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-3) < tol ){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }
    else if ( (iter == maxiter) & (abs(ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-3) >= tol) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("ll" = ll, "betas" = betas[iter,]))
}






