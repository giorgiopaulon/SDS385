
# Functions used to solve the linear system
# Beta_hat = (X^T %*% W %*% X)^(-1) X^T %*% W %*% Y
# whose solution minimizes 
# 1/2*(Y - X %*% Beta)^T %*% W %*% (Y - X%*%Beta)


# Optimal function calculating WLS via inverting the matrix and optimizing the 
# product with the diagonal matrix
my_inv <- function(X, y, W){
  beta_hat <- solve(t(X*diag(W))%*%X)%*%(t(X*diag(W))%*%y)
  # This is much more efficient than 
  # beta_hat <- solve(t(X)%*%W%*%X)%*%(t(X)%*%W%*%y)
  # because we avoid the matricial product by the diagonal matrix W by simply 
  # multiplying the first column by the first element of the diagonal, and so on...
  return(beta_hat)
}

# Optimal function calculating WLS via Cholesky decomposition and optimizing the 
# product with the diagonal matrix
my_chol <- function(X, y, W){
  inverse <- chol2inv(chol(t(X*diag(W))%*%X))
  # This is much more efficient than 
  # inverse <- chol2inv(chol(t(X)%*%W%*%X))
  # because we avoid the matricial product by the diagonal matrix W by simply 
  # multiplying the first column by the first element of the diagonal, and so on...
  beta_hat <- inverse%*%(t(X*diag(W))%*%y) 
  return(beta_hat)
}

# Optimal function calculating WLS via LU factorization and optimizing the 
# product with the diagonal matrix
my_LU <- function(X, y, W){
  LU <- expand(lu(t(X*diag(W))%*%X))
  U <- LU$U
  L <- LU$L
  b <- solve(L, t(X*diag(W))%*%y)
  beta_hat <- solve(U, b)
  return(beta_hat)
}