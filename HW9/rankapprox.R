

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

l1.norm <- function(vec){
  # Computes the l1 norm of a vector
  # --------------------------------
  # Args: 
  #   - vec: vector whose norm has to be computed
  # Returns: 
  #   - the l1 norm of the vector
  # -----------------------------
  return(sum(abs(vec)))
}

l2.norm <- function(vec){
  # Computes the l2 norm of a vector
  # --------------------------------
  # Args: 
  #   - vec: vector whose norm has to be computed
  # Returns: 
  #   - the l2 norm of the vector
  # -----------------------------
  return(sqrt(sum(vec^2)))
}


BinarySearch <- function(argu,sumabs){
  # Computes the soft thresholding parameter via Binary Search
  # ----------------------------------------------------------
  # Args: 
  #   - argu: vector whose soft thresholding has to be computed
  #   - sumabs: l1 norm of the vector
  # Returns: 
  #   - the soft thresholding parameter
  # -----------------------------------
  if(l2.norm(argu)==0 || sum(abs(argu/l2.norm(argu)))<=sumabs) return(0)
  lam1 <- 0
  lam2 <- max(abs(argu))-1e-5
  iter <- 1
  while(iter < 150){
    su <- soft.thresholding(argu,(lam1+lam2)/2)
    if(sum(abs(su/l2.norm(su)))<sumabs){
      lam2 <- (lam1+lam2)/2
    } else {
      lam1 <- (lam1+lam2)/2
    }
    if((lam2-lam1)<1e-6) return((lam1+lam2)/2)
    iter <- iter+1
  }
  warning("Didn't quite converge")
  return((lam1+lam2)/2)
}


PMD <- function(X, c1, c2, maxiter = 1000, tol = 1E-6){
  # Computes the penalized matrix decomposition of rank 1
  # -----------------------------------------------------
  # Args: 
  #   - X: matrix whose rank-one approximation has to be computed
  #   - c1: sparsity level for u
  #   - c2: sparsity level for v
  #   - maxiter: maximum number of iterations
  #   - tol: tolerance for the convergence criterion
  # Returns: 
  #   - X.app: the approximated X matrix
  #   - u: 
  #   - d: 
  #   - v: 
  # -----------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  v <- rep(1, p)
  v <- v / l2.norm(v)
  u <- rep(1, n)
  
  for (iter in 1:maxiter){
    
    v.old <- v
    u.old <- u
    
    # u-update step
    Xv <- X %*% v
    delta1 <- BinarySearch(Xv, c1)
    soft <- as.numeric(soft.thresholding(Xv, delta1))
    u <- soft / l2.norm(soft)
    
    # v-update step
    Xtu <- crossprod(X, u)
    delta2 <- BinarySearch(Xtu, c2)
    soft <- as.numeric(soft.thresholding(Xtu, delta2))
    v <- soft / l2.norm(soft)
    
    if(sum(abs(v.old - v)) < tol & sum(abs(u.old - u)) < tol){
      cat('Algorithm has converged after', iter, 'iterations')
      break;
    }
    else if ( (iter == maxiter) & (sum(abs(v.old - v)) >= tol || sum(abs(u.old - u)) >= tol) ){
      print('WARNING: algorithm has not converged')
      break;
    }
  }
  d <- as.numeric(crossprod(u, X) %*% v)
  X.app <- d * tcrossprod(u,v)
  
  return (list(X.app=X.app, u=u, d=d, v=v))
}
