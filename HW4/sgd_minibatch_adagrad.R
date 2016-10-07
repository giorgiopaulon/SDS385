
comp.wi <- function(X, beta){
  # Computes the probabilities associated with the logit model
  # ----------------------------------------------------------
  # Args: 
  #   - X: matrix of the features (n*p)
  #   - beta: regression parameters (length p)
  # Returns: 
  #   - wi: the weights for each individual in the dataset
  # ------------------------------------------------------
  wi <- 1 / (1 + exp(-X %*% beta))
  return(wi)
}


log.lik <- function(beta, y, X, mi){
  # Computes the negative log-likelihood of the logit model
  # -------------------------------------------------------
  # Args: 
  #   - beta: regression parameters (length p)
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - mi: vector of the number of trials, always 1 in the logit framework (length n)
  # Returns: 
  #   - ll: the scalar associated with the negative log-likelihood of the data
  # --------------------------------------------------------------------------
  ll <- array(NA, dim = N)
  wi <- comp.wi(X, beta)
  ll <- -sum(y * log(wi + 1E-10) + (mi - y) * log(1 - wi + 1E-10))
  return(ll)
}


grad.loglik <- function(beta, y, X, mi){
  # Computes the gradient of negative log-likelihood of the logit model
  # -------------------------------------------------------------------
  # Args: 
  #   - beta: regression parameters (length p)
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - mi: vector of the number of trials, always 1 in the logit framework (length n)
  # Returns: 
  #   - grad.ll: the gradient vector of the negative log-lik of the data (length p)
  # -------------------------------------------------------------------------------
  grad.ll <- array(NA, dim = length(beta))
  wi <- comp.wi(X, beta)
  grad.ll <- - crossprod(X, as.numeric(y - mi * wi))
  return(grad.ll)
}


optimal.step <- function(param, direct, grad.ll.act, y, X, mi, c=0.01, max.alpha=1, rho=0.5){
  # Computes the line search
  # ------------------------
  # Args: 
  #   - param: actual beta parameters we are moving from
  #   - direct: chosen descent direction 
  #   - ll.act: actual value of the log-likelihood (we pass it just to save computations)
  #   - grad.ll.act: actual value of the gradient of the log-likelihood (we pass it just to save computations)
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - mi: vector of the number of trials, always 1 in the logit framework (length n)
  #   - c: 
  #   - maxalpha: initial step size (maximum accepted value)
  #   - rho: shrinkage factor
  # Returns: 
  #   - opt.alpha: the optimal step size for the requested direction
  # ----------------------------------------------------------------
  ll.act <- log.lik(param, y, X, mi)
  
  opt.alpha <- max.alpha
  while(log.lik(param + opt.alpha * direct, y, X, mi) > ll.act + c * opt.alpha * crossprod(grad.ll.act, direct)){
    opt.alpha <- opt.alpha * rho
  }
  return(opt.alpha)
}


SGD.linesearch <- function(y, X, mi, beta0, maxiter = 100000, tol = 1E-8){
  # Function for the gradient descent of the logit model coupled with backtracking
  # ------------------------------------------------------------------------------
  # Args: 
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - mi: vector of the number of trials, always 1 in the logit framework (length n)
  #   - beta0: initial regression parameters (length p)
  #   - maxiter: number of maximum iterations the algorithm will perform if not converging
  #   - tol: tolerance threshold to convergence
  # Returns: 
  #   - ll: values of the log-likelihood for every iteration
  #   - beta: final optimal regression parameters 
  #   - alpha: selected step size for every iteration
  # -------------------------------------------------
  N <- dim(X)[1]
  P <- length(beta0)
  
  # Select the updating time and the size of the minibatches 
  train.epoch <- floor(maxiter/1000)
  size.minibatch <- floor(N/10)

  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)
  # Sample the indexes for the minibatch
  minibatches <- matrix(sample(1:N, 1000*size.minibatch, replace = TRUE), 1000, size.minibatch)
  idx.minibatch <- minibatches[1, ]
  
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1, ] <- beta0 # Initial guess
  
  av.ll <- array(NA, dim = maxiter)
  ll <- log.lik(betas[1, ], y[idx[1]], matrix(X[idx[1],], nrow=1), mi[idx[1]])
  av.ll[1] <- ll
  
  for (iter in 2:maxiter){
    
    # Choose the direction according to the gradient of the single individual idx[i]
    grad <- grad.loglik(betas[iter-1,], y[idx[iter]], matrix(X[idx[iter], ],nrow=1), mi[idx[iter]])
    dir <- - grad
    # if we need to update the optimal step
    if ((iter == 2) | (iter %% train.epoch == 0)){
      idx.minibatch <- minibatches[ceiling(iter/train.epoch), ]
      grad.minibatch <- grad.loglik(betas[iter-1, ], y[idx.minibatch], X[idx.minibatch, ], mi[idx.minibatch]) / length(idx.minibatch)
      alpha <- optimal.step(betas[iter-1, ], -grad.minibatch, grad.minibatch, y[idx[iter]], matrix(X[idx[iter], ], nrow=1), mi[idx[iter]])
    }

    # Update Beta parameters
    betas[iter, ] <- betas[iter-1, ] + alpha * dir
    # Compute log-likelihood and average log-likelihood
    ll <- log.lik(betas[iter, ], y[idx[iter]], matrix(X[idx[iter],], nrow=1), mi[idx[iter]])
    av.ll[iter] <- ((av.ll[iter-1])*(iter-1) + ll)/iter
    
    # Convergence check
    if (abs(av.ll[iter-1] - av.ll[iter]) / (av.ll[iter-1] + 1E-10) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      av.ll <- av.ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }

    else if (iter == maxiter & abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("ll" = av.ll, "beta" = betas))
}


AdaGrad <- function(y, X, mi, beta0, maxiter = 100000, tol = 1E-8, eta = 20){
  # Function for Adaptive Gradient Algorithm
  # ----------------------------------------
  # Args: 
  #   - y: response vector (length n)
  #   - X: matrix of the features (n*p)
  #   - mi: vector of the number of trials, always 1 in the logit framework (length n)
  #   - beta0: initial regression parameters (length p)
  #   - maxiter: number of maximum iterations the algorithm will perform if not converging
  #   - tol: tolerance threshold to convergence
  #   - eta: the step size shrinkage factor
  # Returns: 
  #   - av.ll: values of the average of the log-likelihood for every iteration
  #   - betas: regression parameters at every iteration 
  # -------------------------------------------------
  N <- dim(X)[1]
  P <- length(beta0)

  # Sample the data points to calculate the gradient
  idx <- sample(1:N, maxiter, replace = TRUE)

  # Initialize the structures to save parameters
  betas <- array(NA, dim=c(maxiter, length(beta0)))
  betas[1, ] <- beta0 # Initial guess
  # Initialize the structures to save the log-likelihood
  av.ll <- array(NA, dim = maxiter)
  ll <- log.lik(betas[1, ], y[idx[1]], matrix(X[idx[1],], nrow=1), mi[idx[1]])
  av.ll[1] <- ll
  # Initialize the correction G (historical gradient)
  G <- rep(0, P)
  
  for (iter in 2:maxiter){
    
    # Choose the direction according to the gradient of the single individual idx[i], times the correction G
    grad <- grad.loglik(betas[iter-1,], y[idx[iter-1]], matrix(X[idx[iter-1], ],nrow=1), mi[idx[iter-1]])
    G <- G + grad^2 
    dir <- - grad/(sqrt(G) + 1E-8)
    
    # Update Beta parameters
    betas[iter, ] <- betas[iter-1, ] + eta * dir
    # Compute log-likelihood and average log-likelihood
    ll <- log.lik(betas[iter, ], y[idx[iter]], matrix(X[idx[iter],], nrow=1), mi[idx[iter]])
    av.ll[iter] <- ((av.ll[iter-1])*(iter-1) + ll) / iter
    
    # Convergence check
    if (abs(av.ll[iter-1] - av.ll[iter]) / (av.ll[iter-1] + 1E-10) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      av.ll <- av.ll[1:iter]
      betas <- betas[1:iter, ]
      break;
    }

    else if (iter == maxiter & abs(av.ll[iter-1] - av.ll[iter])/(av.ll[iter-1] + 1E-10) >= tol){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("ll" = av.ll, "beta" = betas))
}
