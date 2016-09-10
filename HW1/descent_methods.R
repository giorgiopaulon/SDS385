
# DESCENT METHODS

# Compute the probabilities associated with the logit model
comp_wi <- function(X, beta){
  wi <- 1/(1+exp(-X%*%beta))
  return(wi)
}

# Compute the negative log-likelihood of the logit model
log_lik <- function(beta, y, X, mi){
  ll <- array(NA, dim = N)
  wi <- comp_wi(X, beta)
  ll <- -sum(y*log(wi+1E-4) + (mi - y)*log(1-wi+1E-4))
  return(ll)
}

# Compute the gradient of the negative log-likelihood of the logit model
grad_loglik <- function(beta, y, X, mi){
  grad_ll <- array(NA, dim = length(beta))
  wi <- comp_wi(X, beta)
  grad_ll <- -apply(X*as.numeric(y - mi*wi), 2, sum)
  return(grad_ll)
}

# Compute the hessian of the negative log-likelihood of the logit model
hessian_loglik <- function(beta, y, X, mi){
  wi <- comp_wi(X, beta)
  W_12 <- sqrt(mi*wi*(1-wi))
  hes_ll <- crossprod(as.numeric(W_12)*X)
  return(hes_ll)
}

# Function for the gradient descent of the logit model
gradient_descent <- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    
    gradient <- grad_loglik(betas[iter-1,], y, X, mi)
    betas[iter,] <- betas[iter-1,] - alpha*gradient
    ll[iter] <- log_lik(betas[iter,], y, X, mi)
    
    # Convergence check
    if ((ll[iter-1] - ll[iter])/(ll[iter-1] + 1E-3) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }
    
    else if (iter == maxiter & (ll[iter-1] - ll[iter])/(ll[iter-1] + 1E-2) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter,]))
}

# Function for the Newton method of the logit model
newton_descent <- function(y, X, mi, beta0, maxiter, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    
    hessian <- hessian_loglik(betas[iter-1,], y, X, mi)
    gradient <- grad_loglik(betas[iter-1,], y, X, mi)
    
    # Solve the linear system H^(-1)%*%Grad in order to find the direction
    direct <- QR_solver(hessian, gradient)
    
    betas[iter,] <- betas[iter-1,] - direct
    ll[iter] <- log_lik(betas[iter,], y, X, mi)
    
    # Convergence check
    if (sqrt(sum((betas[iter,] - betas[iter-1,])^2)) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }
    
    else if (iter == maxiter & sqrt(sum((betas[iter,] - betas[iter-1,])^2)) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter,]))
}
