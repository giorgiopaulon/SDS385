
# Functions used to solve the linear system
# Beta_hat = (X^T %*% W %*% X)^(-1) X^T %*% W %*% Y
# whose solution minimizes 
# 1/2*(Y - X %*% Beta)^T %*% W %*% (Y - X%*%Beta)


# Optimal function calculating WLS via inverting the matrix and optimizing the 
# product with the diagonal matrix
my_inv <- function(X, y, W){
  beta_hat <- solve(crossprod(diag(W)^(1/2)*X))%*%(t(X*diag(W))%*%y)
  
  return(beta_hat)
}

# Optimal function calculating WLS via Cholesky decomposition and optimizing the 
# product with the diagonal matrix
my_chol <- function(X, y, W){
  inverse <- chol2inv(chol(crossprod(diag(W)^(1/2)*X)))
  beta_hat <- inverse%*%(t(X*diag(W))%*%y) 
  
  return(beta_hat)
}

# Optimal function calculating WLS via QR factorization
my_QR <- function(X, y, W){
  P <- ncol(X)
  QR <- qr(diag(W)^(1/2)*X)
  Q <- qr.Q(QR)
  R <- qr.R(QR)
  b <- t(Q*diag(W)^(1/2))%*%y
  beta_hat <- backsolve(R, b)

  return(beta_hat)
}

# Optimal function calculating WLS via inverting the matrix and exploiting the
# sparsity of the matrix X
my_inv_sparse <- function(X, y, W){
  X = Matrix(X, sparse=TRUE)
  beta_hat <- solve(crossprod(diag(W)^(1/2)*X))%*%(t(X*diag(W))%*%y)
  
  return(beta_hat)
}

# QR solver for the system Ax = b
QR_solver <- function(A, b){
  QR <- qr(A)
  x <- solve.qr(QR, b)
  
  return(x)
}
