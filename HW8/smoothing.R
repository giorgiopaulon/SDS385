

makeD2_sparse = function (dim1, dim2){
  require(Matrix)
  D1 = bandSparse(dim1 * dim2, m = dim1 * dim2, k = c(0, 1), 
                  diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * 
                                                               dim2 - 1)))
  D1 = D1[(seq(1, dim1 * dim2)%%dim1) != 0, ]
  D2 = bandSparse(dim1 * dim2 - dim1, m = dim1 * dim2, k = c(0, 
                                                             dim1), diagonals = list(rep(-1, dim1 * dim2), rep(1,                                                                                               dim1 * dim2 - 1)))
  return(rBind(D1, D2))
}


laplacian.direct <- function(y, L, lambda){
  # Direct solver for the laplacian smoothing
  # -----------------------------------------
  # Args:
  #   - y: response vector of the observations (length n)
  #   - L: laplacian matrix
  #   - lambda: penalization parameter for the laplacian (threshold)
  # Returns:
  #   - x: values of the mean functional on the grid
  # ------------------------------------------------
  
  # Compute the matrix H of the linear system Hx = y
  H <- lambda * L
  diag(H) <- diag(H) + 1
  
  # Solve the linear system
  x <- Matrix::solve(H, y)
  # R = chol(H)
  # u = forwardsolve(t(R), y)
  # x = backsolve(R, u)
  
  return (x)
}


laplacian.jacobi <- function(y, L, lambda, maxiter){
  # Jacobi iterative solver for the laplacian smoothing
  # ---------------------------------------------------
  # Args:
  #   - y: response vector of the observations (length n)
  #   - L: laplacian matrix
  #   - lambda: penalization parameter for the laplacian (threshold)
  #   - maxiter: maximum of iterations allowed
  # Returns:
  #   - x: values of the mean functional on the grid
  #   - LOOCV: estimate of the leave-one-out cross validation error
  # ---------------------------------------------------------------
  
  # Compute the decomposition of the matrix H of the linear system Cx = y
  # where C = lambda * L + I
  D <- lambda * diag(L) + 1
  R <- lambda * L
  diag(R) <- rep(0, length(D))
  
  # Solve the linear system
  x <- rep(0, length(y))
  for (iter in 1:maxiter){
    xold <- x
    x <- (1/D)*(y - R %*% xold)
    if (max(abs(x - xold)) < 1E-3){
      break;
    }
    else if (max(abs(x - xold)) >= 1E-3 & iter == maxiter){
      cat("WARNING: Jacobi Method did not converge\n")
      break;
    }
  }

  return (x)
}


laplacian.gaussseidel <- function(y, L, lambda, maxiter){
  # Jacobi iterative solver for the laplacian smoothing
  # ---------------------------------------------------
  # Args:
  #   - y: response vector of the observations (length n)
  #   - L: laplacian matrix
  #   - lambda: penalization parameter for the laplacian (threshold)
  #   - maxiter: maximum of iterations allowed
  # Returns:
  #   - x: values of the mean functional on the grid
  #   - LOOCV: estimate of the leave-one-out cross validation error
  # ---------------------------------------------------------------
  
  # Compute the decomposition of the matrix H of the linear system Cx = y
  # where C = lambda * L + I
  A <- lambda * L
  diag(A) <- diag(A) + 1
  
  U <- triu(A, k = 1)
  L.star = tril(A)
  
  # Solve the linear system
  x <- rep(0, length(y))
  for (iter in 1:maxiter){
    xold <- x
    x <- Matrix::solve(L.star, y - U %*% x, system = 'L')
    if (max(abs(x - xold)) < 1E-3){
      break;
    }
    else if (max(abs(x - xold)) >= 1E-3 & iter == maxiter){
      cat("WARNING: Gauss-Seidel Method did not converge\n")
      break;
    }
  }
  
  return (x)
}


objective.l2 <- function(x, z, y, L, lambda){
  # Computes the objective function with L2 penalty
  # -----------------------------------------------
  # Args: 
  #   - x: actual point
  #   - z: actual point
  #   - y: response vector of the data
  #   - L: laplacian penalty matrix
  #   - lambda: penalization parameter
  # Returns: 
  #   - obj: the scalar associated with the objective function at beta
  # ------------------------------------------------------------------
  obj <- (1/2)*crossprod(x - y) + (lambda/2) * crossprod(z, L %*% z)
  return(obj)
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


objective.l1 <- function(x, z, y, L, lambda){
  # Computes the objective function with L1 penalty
  # -----------------------------------------------
  # Args: 
  #   - x: actual point
  #   - z: actual point
  #   - y: response vector of the data
  #   - L: laplacian penalty matrix
  #   - lambda: penalization parameter
  # Returns: 
  #   - obj: the scalar associated with the objective function at beta
  # ------------------------------------------------------------------
  obj <- (1/2)*crossprod(x - y) + (lambda/2) * crossprod(z)
  return(obj)
}


fusedlasso.admm <- function(y, D, x0, lambda, rho = 1, maxiter = 5000, tol.abs = 1E-3, tol.rel = 1E-3){
# ADMM that allows to compute the Fused Lasso
# -------------------------------------------
# Args:
#   - y: response vector of the observations (length n)
#   - D: oriented edge matrix of the graph
#   - x0: initial values of x
#   - lambda: penalization parameter for the lasso (threshold)
#   - rho: step size
#   - maxiter: maximum number of iterations
#   - tol.abs: tolerance defining convergence
#   - tol.rel: tolerance defining convergence
# Returns:
#   - obj: values of the objective function for each iteration
#   - x: values of the mean functional on the grid
# ------------------------------------------------

  L <- crossprod(D)

  # Check if the inputs have the correct sizes
  if ((length(y) != dim(L)[1]) || (length(x0) != dim(L)[1])){
    stop("Incorrect input dimensions.")
  }

  n <- length(x0)
  m <- dim(D)[1]

  # Initialize the data structures
  x <- array(NA, dim = n)
  x <- x0 # Initial guess
  r <- array(NA, dim = m)
  r <- rep(0, m)
  u <- array(NA, dim = m)
  u <- rep(0, m)
  obj <- array(NA, dim = maxiter)
  obj[1] <- objective.l1(x, r, y, L, lambda)

  # Precaching operations
  E <- rho * L
  diag(E) <- diag(E) + 1

  for (iter in 2:maxiter){

    rold <- r

    # x - minimization step
    x <- Matrix::solve(E, y + rho * crossprod(D, r - u))

    # r - minimization step
    Dx <- D %*% x
    r <- soft.thresholding(as.vector(Dx + u), lambda/rho)

    # u - minimization step
    q <- Dx - r
    u <- u + q

    # Compute log-likelihood
    obj[iter] <- objective.l1(x, r, y, L, lambda)

    rnorm <- sqrt(sum(r^2))
    snorm <- rho * sqrt(sum((r - rold)^2))
    eprim <- sqrt(n) * tol.abs + tol.rel * max(c(sqrt(sum(x^2)), sqrt(sum(r^2)), 0))
    edual <- sqrt(n) * tol.abs + tol.rel * rho * sqrt(sum(u^2))
    # Convergence check
    if ( (rnorm < eprim) & (snorm < edual) ){
      cat('Algorithm has converged after', iter, 'iterations')
      obj <- obj[1:iter]
      break;
    }
    else if ( (iter == maxiter) & ((rnorm >= eprim) || (snorm >= edual)) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("obj" = obj, "x" = x))
}


fusedlasso.refined <- function(y, D, x0, lambda, rho = 1, maxiter = 5000, tol.abs = 1E-5, tol.rel = 1E-5){
  # Optimized ADMM that allows to compute the Fused Lasso
  # -----------------------------------------------------
  # Args:
  #   - y: response vector of the observations (length n)
  #   - D: oriented edge matrix of the graph
  #   - x0: initial values of x
  #   - lambda: penalization parameter for the lasso (threshold)
  #   - rho: step size
  #   - maxiter: maximum number of iterations
  #   - tol.abs: tolerance defining convergence
  #   - tol.rel: tolerance defining convergence
  # Returns:
  #   - obj: values of the objective function for each iteration
  #   - x: values of the mean functional on the grid
  # ------------------------------------------------
  
  L <- crossprod(D)
  
  # Check if the inputs have the correct sizes
  if ((length(y) != dim(L)[1]) || (length(x0) != dim(L)[1])){
    stop("Incorrect input dimensions.")
  }
  
  # Rescaling the threshold to make it comparable to glmnet
  # lambda <- lambda * length(y)
  n <- length(x0)
  m <- dim(D)[1]
  
  # Initialize the data structures
  x <- array(NA, dim = n)
  x <- x0 # Initial guess
  z <- array(NA, dim = n)
  z <- rep(0, n)
  u <- array(NA, dim = n)
  u <- rep(0, n)
  t <- array(NA, dim = m)
  t <- rep(0, m)
  s <- array(NA, dim = m)
  s <- rep(0, m)
  obj <- array(NA, dim = maxiter)
  obj[1] <- objective.l1(z, z, y, L, lambda)
  
  # Precaching operations
  E <- L
  diag(E) <- diag(E) + 1
  #diag(E) <- diag(E) + rho

  for (iter in 2:maxiter){
    
    zold <- z
    
    # x - minimization step
    x <- (as.vector(y) + rho * (z - u)) / (1 + rho)
    
    # r - minimization step
    r <- soft.thresholding(as.vector(s - t), lambda)
    #r <- soft.thresholding(as.vector(s - t), lambda/rho)
    
    # (z, s) - minimization step
    Dtv <- crossprod(D, r + t)
    z <- Matrix::solve(E, x + u + Dtv, sparse = T)
    s <- D %*% z
    
    # (u, t) - minimization step
    res <- x - z
    u <- u + res
    t <- t + r - s
    
    # Compute log-likelihood
    obj[iter] <- objective.l1(x, x, y, L, lambda)
    
    rnorm <- sqrt(sum(res^2))
    snorm <- rho * sqrt(sum((z - zold)^2))
    eprim <- sqrt(n) * tol.abs + tol.rel * max(c(sqrt(sum(x^2)), sqrt(sum(z^2)), 0))
    edual <- sqrt(n) * tol.abs + tol.rel * rho * sqrt(sum(u^2))
    # Convergence check
    if ( (rnorm < eprim) & (snorm < edual) ){
      cat('Algorithm has converged after', iter, 'iterations')
      obj <- obj[1:iter]
      break;
    }
    else if ( (iter == maxiter) & ((rnorm >= eprim) || (snorm >= edual)) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  return(list("obj" = obj, "x" = x))
}