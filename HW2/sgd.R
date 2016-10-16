

SGD <- function(y, X, mi, beta0, maxiter, alpha, tol){
  
  N <- dim(X)[1]
  
  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  av.ll <- array(NA, dim = maxiter)
  av.ll[1] <- log.lik(betas[1,], matrix(y[idx[1]],nrow=1), matrix(X[idx[1],],nrow=1), mi[idx[1]])
  
  for (iter in 2:maxiter){
    
    gradient <- grad.loglik(betas[iter-1,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    betas[iter,] <- betas[iter-1,] - alpha*gradient
    ll <- log.lik(betas[iter,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    av.ll[iter] <- ((av.ll[iter-1])*(iter-1) + ll)/iter

    # Convergence check
    if (abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      av.ll <- av.ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }

    else if (iter == maxiter & abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
    
  }
  
  return(list("ll" = av.ll, "beta" = betas))
}


# Function for the gradient descent of the logit model
SGD.Robbins <- function(y, X, mi, beta0, maxiter, C, t0, alpha, tol){
  
  N <- dim(X)[1]
  
  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1,] <- beta0 # Initial guess
  
  av.ll <- array(NA, dim = maxiter)
  av.ll[1] <- log.lik(betas[1,], matrix(y[idx[1]],nrow=1), matrix(X[idx[1],],nrow=1), mi[idx[1]])
  
  for (iter in 2:maxiter){
    
    step.size <- C*(iter+t0)^(-alpha)
    
    gradient <- grad.loglik(betas[iter-1,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    betas[iter,] <- betas[iter-1,] - step.size*gradient
    ll <- log.lik(betas[iter,], matrix(y[idx[iter]],nrow=1), matrix(X[idx[iter],],nrow=1), mi[idx[iter]])
    av.ll[iter] <- ((av.ll[iter-1])*(iter-1) + ll)/iter
    
    # Convergence check
    if (abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      av_ll <- av_ll[1:iter]
      betas <- betas[1:iter,]
      break;
    }

    else if (iter == maxiter & abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = av.ll, "beta" = betas))
}
