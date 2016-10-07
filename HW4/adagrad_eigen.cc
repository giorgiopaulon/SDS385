
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
SEXP Ada_Grad(MapMatd& X, VectorXd& y, VectorXd& m, VectorXd& beta0, double eta = 1.0, unsigned int npass = 1, double lambda = 0.0, double weight = 0.01){
  // - X: 		design matrix stored in sparse column-major format. That is, we transposed the matrix outside the function (in R).
  //   			The features for the observation i is stored in column i
  // - y: 		response vector
  // - m: 		vector of sample sizes
  // - beta0: 	initial guess values for beta
  // - eta: 	master step-size
  // - npass: 	number of times we go over the dataset
  // - lambda: 	penalization of the Lasso regression (L1 penalty)
  // - weight: 	weight for the computation of the negative log-likelihood (high weight gives importance at every single contribution, small weight smoothes the
  //			negative log-likelihood function).
  
  unsigned int N = X.cols();
  unsigned int P = X.rows();
  
  // x is the generic column of X that will be extracted at each iteration
  SparseVector<double> x(P);
  
  // Initialize parameters
  double psi, epsi, yhat, delta, h;
  double this_grad = 0.0;
  double newbeta, penalty;
  unsigned int j,k;

  // Initialize parameters alpha and beta
  double alpha = 0.0;
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
  double loglik_avg = 0.0;
  NumericVector loglik_vec(npass*N, 0.0);
  
  
  // Outer loop: number of passes over data set
  k = 0; // global interation counter
  for(unsigned int pass = 0; pass < npass; pass++) {
    
    // Loop over each observation (columns of X)
    for(unsigned int i = 0; i < N; i++) {
      
      // Compute linear predictor and yhat
      x = X.innerVector(i); 			// efficient way to extract a column from the matrix X
      psi = alpha + x.dot(beta); 		// the function .dot() automatically ignores the zero elements of x
      epsi = exp(psi);
      yhat = m(i) * epsi/(1.0 + epsi);	// MLE of y
	        
      // (1) Update log-likelihood weighted average
      loglik_avg = (1.0 - weight) * loglik_avg + weight * (m(i) * log(1 + epsi) - y(i) * psi);
      loglik_vec(k) = loglik_avg;
      
      // (2) Update intercept
      delta = y(i) - yhat; 						// GRADIENT with respect to the intercept (the intercept component)
      hist_int += delta * delta; 				// update historical gradient of the intercept
      alpha += (eta/sqrt(hist_int)) * delta;	// compute the update of the intercept  
	  	  
      // (3) Update beta: iterate over the ACTIVE features for this instance
      for (SparseVector<double>::InnerIterator it(x); it; ++it) {
        
        // the .index() function extracts the x index of iterator currently pointing at x
        j = it.index();
        
        // STEP (a): aggregate all the penalty-only updates since the last time we updated this feature.
        // This is a form of lazy updating in which we approximate all the "penalty-only" updates at once.
        double skip = k - last_update(j); 		// how many penalization updates did we skip for this feature? 
        h = sqrt(hist_grad(j));
        penalty = skip * eta/h;				// penalize the skip previous updates
		
		// We can find the corresponding beta applying the penalty. Let us remark that if beta has changed sign, we set it to 0. This is reasonable 
        beta(j) = sgn(beta(j)) * fmax(0.0, fabs(beta(j)) - penalty*lambda);
		// For the L2 penalty, we can change this line to: 2*lambda*skip*beta(j), which is an approximation because beta is changing at each iteration
				
        // Update the last-update vector
        last_update(j) = k;
        
        // STEP (b): Now we compute the regular update for this observation.
        this_grad = delta*it.value();
        
        // update the historical gradient for this feature
        hist_grad(j) += this_grad*this_grad;
        
        h = sqrt(hist_grad(j));
		penalty = eta/h;
		newbeta = beta(j) + penalty * this_grad;
        beta(j) = sgn(newbeta) * fmax(0.0, fabs(newbeta) - penalty * lambda);		
      }
      k++; // increment global counter
    }
  }
  
  // (4) Update last penalties:
  // At the very end, apply the accumulated penalty for the variables we haven't touched recently. We still have to penalize some features that we did not update 
  // during the last "skip" iterations.
  for (int j = 0; j < P; j++) {
    double skip = k - last_update(j);
    h = sqrt(hist_grad(j));
    penalty = skip*eta/h;
    beta(j) = sgn(beta(j))*fmax(0.0, fabs(beta(j)) - penalty*lambda);
  }
  
  return List::create(Named("alpha") = alpha,
                      Named("beta") = beta, 
					  Named("loglik") = loglik_vec);
}





