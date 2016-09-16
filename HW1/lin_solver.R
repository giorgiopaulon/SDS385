
# Functions used to solve the linear system
# Beta.hat = (X^T %*% W %*% X)^(-1) X^T %*% W %*% Y
# whose solution minimizes 
# 1/2*(Y - X %*% Beta)^T %*% W %*% (Y - X%*%Beta)


# Optimal function calculating WLS via inverting the matrix and optimizing the 
# product with the diagonal matrix
my.inv <- function(X, y, W){
  wx <- diag(W)^(1/2) * X
  wy <- diag(W)^(1/2) * y
  beta.hat <- solve(crossprod(wx), crossprod(wx, wy))

  return(beta.hat)
}

# Optimal function calculating WLS via Cholesky decomposition and optimizing the 
# product with the diagonal matrix
my.chol <- function(X, y, W){
  wx <- diag(W)^(1/2) * X
  wy <- diag(W)^(1/2) * y
  R = chol(crossprod(wx))
  u = forwardsolve(t(R), crossprod(wx, wy))
  beta.hat = backsolve(R, u)
  
  return(beta.hat)
}

# Optimal function calculating WLS via QR factorization
my.QR <- function(X, y, W){
  P <- ncol(X)
  wx <- diag(W)^(1/2) * X
  wy <- diag(W)^(1/2) * y
  QR <- qr(wx)

  qty = qr.qty(QR, wy)
  beta.hat = backsolve(QR$qr, qty)

  return(beta.hat)
}

# Optimal function calculating WLS via inverting the matrix and exploiting the
# sparsity of the matrix X
my.invsparse <- function(X, y, W){
  X = Matrix(X, sparse=TRUE)
  wx <- diag(W)^(1/2) * X
  wy <- diag(W)^(1/2) * y
  beta.hat <- solve(crossprod(wx), crossprod(wx, wy))
  
  return(beta.hat)
}

# Optimal function calculating WLS via Cholesky decomposition and exploiting the
# sparsity of the matrix X
my.cholsparse <- function(X, y, W){
  X = Matrix(X, sparse=TRUE)
  wx <- diag(W)^(1/2) * X
  wy <- diag(W)^(1/2) * y
  R = chol(crossprod(wx))
  u = forwardsolve(t(R), crossprod(wx, wy))
  beta.hat = backsolve(R, u)
  
  return(beta.hat)
}

# VEDERE GLI SPARSE METHODS

# Non usare QR perché calcolare la matrice ortogonale è costoso, inoltre la matrice Q
# è densa!


# QR solver for the system Ax = b
QR.solver <- function(A, b){
  QR <- qr(A)
  x <- solve.qr(QR, b)
  
  return(x)
}
