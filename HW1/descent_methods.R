
# DESCENT METHODS

# Compute the probabilities associated with the logit model
comp.wi <- function(X, beta){
  wi <- 1/(1+exp(-X %*% beta))
  return(wi)
}

# Compute the negative log-likelihood of the logit model
log.lik <- function(beta, y, X, mi){
  N <- length(y)
  ll <- array(NA, dim = N)
  wi <- comp.wi(X, beta)
  ll <- -sum(y * log(wi + 1E-4) + (mi - y) * log(1 - wi + 1E-4))
  return(ll)
}

# Compute the gradient of the negative log-likelihood of the logit model
grad.loglik <- function(beta, y, X, mi){
  grad.ll <- array(NA, dim = length(beta))
  wi <- comp.wi(X, beta)
  grad.ll <- -apply(X * as.numeric(y - mi * wi), 2, sum)
  return(grad.ll)
}

# Compute the hessian of the negative log-likelihood of the logit model
hessian.loglik <- function(beta, y, X, mi){
  wi <- comp.wi(X, beta)
  W.12 <- sqrt(mi * wi * (1 - wi))
  hes.ll <- crossprod(as.numeric(W.12) * X)
  return(hes.ll)
}

# Function for the gradient descent of the logit model
gradient.descent <- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    
    gradient <- grad.loglik(betas[iter-1,], y, X, mi)
    betas[iter,] <- betas[iter-1,] - alpha * gradient
    ll[iter] <- log.lik(betas[iter,], y, X, mi)
    
    # Convergence check
    if ((ll[iter-1] - ll[iter])/(ll[iter-1] + 1E-3) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }
    
    else if (iter == maxiter & (ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-2) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter,]))
}

# Function for the Newton method of the logit model
newton.descent <- function(y, X, mi, beta0, maxiter, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    
    hessian <- hessian.loglik(betas[iter-1,], y, X, mi)
    gradient <- grad.loglik(betas[iter-1,], y, X, mi)
    
    # Solve the linear system H^(-1)%*%Grad in order to find the direction
    direct <- QR.solver(hessian, gradient)
    
    betas[iter,] <- betas[iter-1,] - direct
    ll[iter] <- log.lik(betas[iter,], y, X, mi)
    
    # Convergence check
    if ((ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-3) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }
    
    else if (iter == maxiter & (ll[iter-1] - ll[iter]) / (ll[iter-1] + 1E-2) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter,]))
}
