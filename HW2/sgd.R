

# Function for the SGD of the logit model
SGD <- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  N <- dim(X)[1]
  
  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], matrix(y[idx[1]],nrow=1), matrix(X[idx[1],],nrow=1), mi[idx[1]])
  
  for (iter in 2:maxiter){
    
    gradient <- grad_loglik(betas[iter-1,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    betas[iter,] <- betas[iter-1,] - alpha*gradient
    ll[iter] <- log_lik(betas[iter,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    
#     # Convergence check
#     if (sqrt(sum((betas[iter,] - betas[iter-1,])^2)) < tol){
#       cat('Algorithm has converged after', iter, 'iterations')
#       ll <- ll[1:iter]
#       betas <- betas[1:iter,]
#       break;
#     }
#     
#     else if (iter == maxiter & sqrt(sum((betas[iter,] - betas[iter-1,])^2)) >= tol){
#       print('WARNING: algorithm has not converged')
#       break;
#    }
  }
  
  ll <- cumsum(ll)/(1:iter) # Simple moving average
  
  return(list("ll" = ll, "beta" = betas))
}


# Function for the gradient descent of the logit model
SGD_Robbins <- function(y, X, mi, beta0, maxiter, C, t0, alpha, tol){
  
  N <- dim(X)[1]
  
  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log_lik(betas[1,], matrix(y[idx[1]],nrow=1), matrix(X[idx[1],],nrow=1), mi[idx[1]])
  
  for (iter in 2:maxiter){
    
    step_size <- C*(iter+t0)^(-alpha)
    
    gradient <- grad_loglik(betas[iter-1,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    betas[iter,] <- betas[iter-1,] - step_size*gradient
    ll[iter] <- log_lik(betas[iter,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    
#     # Convergence check
#     if (sqrt(sum((betas[iter,] - betas[iter-1,])^2)) < tol){
#       cat('Algorithm has converged after', iter, 'iterations')
#       ll <- ll[1:iter]
#       betas <- betas[1:iter,]
#       break;
#     }
#     
#     else if (iter == maxiter & sqrt(sum((betas[iter,] - betas[iter-1,])^2)) >= tol){
#       print('WARNING: algorithm has not converged')
#       break;
#     }
   }
  
  ll <- cumsum(ll)/(1:iter) # Simple moving average
  
  return(list("ll" = ll, "beta" = betas))
}

