
#define ARMA_64BIT_WORD
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <iostream>
#include <algorithm>
#include <string>

using namespace Rcpp;

//using Eigen::Map;
//using Eigen::MatrixXd;
//using Eigen::MatrixXi;
using Eigen::VectorXd;
//using Eigen::VectorXi;
using Eigen::SparseVector;

typedef Eigen::MappedSparseMatrix<double>  MapMatd;
// typedef Map<MatrixXi>  MapMati;
// typedef Map<VectorXd>  MapVecd;
// typedef Map<VectorXi>  MapVeci;


// Function to compute the sign of a generic type
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}


// [[Rcpp::export]]
SEXP Ada_Grad(MapMatd& X, VectorXd& y, VectorXd& m, VectorXd& beta0, double eta = 1.0, unsigned int npass = 1, double lambda = 0.0, double discount = 0.01){
  // X is the design matrix stored in sparse column-major format
  // i.e. with features for case i stores in column i
  // y is the response vector
  // m is the vector of sample sizes
  
  unsigned int N = X.cols();
  unsigned int P = X.rows();
  
  // bookkeeping variables
  SparseVector<double> x(P);
  unsigned int j,k;
  
  // Initialize parameters
  double psi0, epsi, yhat, delta, h;
  double this_grad = 0.0;
  double mu, gammatilde;

  // Initialize parameters alpha and beta
  double w_hat = (y.sum() + 1.0) / (m.sum() + 2.0);
  double alpha = log(w_hat/(1.0-w_hat));
  VectorXd beta(P);
  
  // Initialize historical gradients (cumulative)
  double hist_int = 0.0; // historical gradient of the intercept term
  VectorXd hist_grad(P);
  for (int j = 0; j < P; j++){
    hist_grad(j) = 1e-3;
    beta(j) = beta0(j);
  }
  
  // Bookkeeping: how long has it been since the last update of each feature?
  NumericVector last_update(P, 0.0);
  
  // negative log likelihood for assessing fit
  double nll_avg = 0.0;
  NumericVector nll_tracker(npass*N, 0.0);
  
  
  // Outer loop: number of passes over data set
  k = 0; // global interation counter
  for(unsigned int pass = 0; pass < npass; pass++) {
    
    // Loop over each observation (columns of X)
    for(unsigned int i = 0; i < N; i++) {
      
      // Form linear predictor and E(Y[i]) from features
      x = X.innerVector(i);
      psi0 = alpha + x.dot(beta);
      epsi = exp(psi0);
      yhat = m[i] * epsi/(1.0 + epsi);
	        
      // (1) Update nll average
      nll_avg = (1.0 - discount) * nll_avg + discount * (m[i] * log(1 + epsi) - y[i] * psi0);
      nll_tracker[k] = nll_avg;
      
      // (2) Update intercept
      delta = y[i] - yhat; // gradient with respect to the intercept
      hist_int += delta * delta; // update historical gradient of the intercept
      alpha += (eta/sqrt(hist_int)) * delta;	  
	  
	  //Rcpp::Rcout << "alpha = " << alpha << std::endl;
	  
      // (3) Update beta: iterate over the active features for this instance
      for (SparseVector<double>::InnerIterator it(x); it; ++it) {
        
        // Which feature is this?
        j = it.index();
        
        // STEP (a): aggregate all the penalty-only updates since the last time we updated this feature.
        // This is a form of lazy updating in which we approximate all the "penalty-only" updates at once.
        double skip = k - last_update(j);
        h = sqrt(hist_grad(j));
        gammatilde = skip * eta/h;
        beta(j) = sgn(beta(j)) * fmax(0.0, fabs(beta(j)) - gammatilde*lambda);
				
        // Update the last-update vector
        last_update(j) = k;
        
        // STEP (b): Now we compute the update for this observation.
        // gradient of negative log likelihood
        this_grad = delta*it.value();
        
        // update adaGrad scaling for this feature
        hist_grad(j) += this_grad*this_grad;
        
        // scaled stepsize
        h = sqrt(hist_grad(j));
		gammatilde = eta/h;
		mu = beta(j) + gammatilde * this_grad;
        beta(j) = sgn(mu) * fmax(0.0, fabs(mu) - gammatilde * lambda);		
		//Rcpp::Rcout << "beta = " << beta(j) << std::endl;
      }
      k++; // increment global counter
    }
  }
  
  // (4) Update last penalties:
  //At the very end, apply the accumulated penalty for the variables we haven't touched recently
  for (int j = 0; j < P; j++) {
    double skip = k - last_update(j);
    h = sqrt(hist_grad(j));
    gammatilde = skip*eta/h;
    beta(j) = sgn(beta(j))*fmax(0.0, fabs(beta(j)) - gammatilde*lambda);
  }
  
  return List::create(Named("alpha") = alpha,
                      Named("beta") = beta);
}




