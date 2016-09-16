
# DESCENT METHODS COUPLED WITH LINE SEARCH

# Compute the probabilities associated with the logit model
comp.wi <- function(X, beta){
  wi <- 1 / (1 + exp(-X %*% beta))
  return(wi)
}

# Compute the negative log-likelihood of the logit model
log.lik <- function(beta, y, X, mi){
  ll <- array(NA, dim = N)
  wi <- comp.wi(X, beta)
  ll <- -sum(y * log(wi + 1E-10) + (mi - y) * log(1 - wi + 1E-10))
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

# Compute the line search
optimal.step <- function(param.act, direct, ll.act, grad.ll.act, y, X, mi, c=0.01, max.alpha=2, rho=0.5){
  opt.alpha <- max.alpha
  while(log.lik(param.act + opt.alpha * direct, y, X, mi) > ll.act + c * opt.alpha * crossprod(grad.ll.act, direct)){
    opt.alpha <- opt.alpha * rho
  }
  return(opt.alpha)
}

# Function for the gradient descent of the logit model
GD.line.search <- function(y, X, mi, beta0, maxiter, tol){
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1, ] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1, ], y, X, mi)
  
  alpha <- array(NA, dim = maxiter)
  
  for (iter in 2:maxiter){
    
    gradient <- grad.loglik(betas[iter-1, ], y, X, mi)
    direct <- -gradient
    alpha[iter-1] <- optimal.step(betas[iter-1, ], direct, ll[iter-1], gradient, y, X, mi)
    betas[iter, ] <- betas[iter-1, ] + alpha[iter-1] * direct
    ll[iter] <- log.lik(betas[iter, ], y, X, mi)
    
    # Convergence check
    if (abs(ll[iter-1] - ll[iter]) / abs(ll[iter-1] + 1E-3) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }
    
    else if (iter == maxiter & abs(ll[iter-1] - ll[iter]) / abs(ll[iter-1] + 1E-3) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter, ], "alpha" = alpha[1:(iter-1)]))
}

# Function for the Newton method of the logit model
quasi.newton <- function(y, X, mi, beta0, maxiter, tol){
  p <- length(beta0)
  
  betas <- array(NA, dim=c(maxiter, p))
  betas[1, ] <- beta0 # Initial guess
  
  ll <- array(NA, dim = maxiter)
  ll[1] <- log.lik(betas[1, ], y, X, mi)
  gradient_new <- grad.loglik(betas[1, ], y, X, mi)
  
  alpha <- array(NA, dim = maxiter)
  
  # We initialize the inverse of the Hessian to the identity matrix
  id.matrix <- diag(rep(1, p))
  Hk <- id.matrix
  
  for (iter in 2:maxiter){
    # The previous gradient is now the old one
    gradient <- gradient_new
    
    # We find the descent direction and we compute the step size
    direct <- - Hk %*% gradient
    alpha[iter-1] <- optimal.step(betas[iter-1, ], direct, ll[iter-1], gradient, y, X, mi)

    # We update the values of the parameters, of the log-likelihood and of the gradient
    sk <- alpha[iter-1] * direct
    betas[iter,] <- betas[iter-1,] + sk
    ll[iter] <- log.lik(betas[iter,], y, X, mi)
    gradient_new <- grad.loglik(betas[iter, ], y, X, mi)
    
    # We update the inverse of the Hessian matrix
    yk <- gradient_new - gradient
    rhok <- as.numeric(1 / crossprod(yk, sk))
    acc <- tcrossprod(sk, yk)
    Hk <- (id.matrix - rhok * acc) %*% Hk %*% (id.matrix - rhok * t(acc)) + rhok * sk %*% t(sk)

    # Convergence check
    if (abs(ll[iter-1] - ll[iter]) / abs(ll[iter-1] + 1E-3) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      ll <- ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }
    
    else if (iter == maxiter & abs(ll[iter-1] - ll[iter]) / abs(ll[iter-1] + 1E-3) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  
  return(list("ll" = ll, "beta" = betas[iter, ], "alpha" = alpha[1:(iter-1)]))
}
