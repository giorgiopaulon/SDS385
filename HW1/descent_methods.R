
# Descent methods

comp_wi <- function(X, beta){
  wi <- 1/(1+exp(-X%*%beta))
  return(wi)
}

log_lik <- function(beta, y, X, mi){
  ll <- array(NA, dim = N)
  wi <- comp_wi(X, beta)
  ll <- -sum(y*log(wi+1E-4) + (mi - y)*log(1-wi+1E-4))
  return(ll)
}

grad_loglik <- function(beta, y, X, mi){
  grad_ll <- array(NA, dim = length(beta))
  wi <- comp_wi(X, beta)
  grad_ll <- -apply(X*as.numeric(y - mi*wi), 2, sum)
  return(grad_ll)
}

gradient_descent<- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    betas[iter,] <- betas[iter-1,] - alpha*grad_loglik(betas[iter-1,], y, X, mi)
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
  
  return(list("ll" = ll, "betas" = betas))
}


newton_descent<- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], y, X, mi)
  
  for (iter in 2:maxiter){
    
    # Compute W
    wi <- comp_wi(X, betas[iter-1,])
    W <- diag(as.numeric(mi*wi*(1-wi)))
    
    # Compute z
    Winv <- diag(as.numeric(1/(mi*wi*(1-wi))))
    z <- Winv%*%(mi*wi - y) - X%*%betas[iter-1,]
    
    # Solve the linear system (X^T*W*X)^-1*X^T*W*z in order to find the direction
    direct <- my_chol(X, z, W)
    
    betas[iter,] <- betas[iter-1,] - alpha*direct
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
  
  return(list("ll" = ll, "betas" = betas))
}