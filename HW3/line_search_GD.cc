#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <armadillo>
#include <string>

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
vec comp_wi(mat X, vec beta){
	// Computes the probabilities associated with the logit model
	// ----------------------------------------------------------
	// Args: 
	//   - X: matrix of the features (n*p)
    //   - beta: regression parameters (length p)
    // Returns:
    //   - wi: the weights for each individual in the dataset
    // ------------------------------------------------------
	// Initialize variables
	unsigned int N = X.n_rows;
	vec wi = zeros(N);
	
	// Compute the probabilities
	wi = 1.0 / (1.0 + exp(-X * beta));

	return wi;
}

// [[Rcpp::export]]
double log_lik(vec beta, vec y, mat X, vec mi){
    // Computes the negative log-likelihood of the logit model
    // -------------------------------------------------------
    // Args:
    //   - beta: regression parameters (length p)
    //   - y: response vector (length n)
    //   - X: matrix of the features (n*p)
    //   - mi: vector of the number of trials, always 1 in the logit framework (length n)
    // Returns:
    //   - ll: the scalar associated with the negative log-likelihood of the data
    // --------------------------------------------------------------------------
	// Initialize variables
	unsigned int N = X.n_rows;
	double ll = 0.0;
	vec wi = zeros(N);
	
	// Compute the log-likelihood
	wi = comp_wi(X, beta);
	ll = - sum(y % log(wi + 1E-10) + (mi - y) % log(1.0 - wi + 1E-10));
	
	return ll;
}

// [[Rcpp::export]]
vec grad_loglik(vec beta, vec y, mat X, vec mi){
    // Computes the gradient of negative log-likelihood of the logit model
    // -------------------------------------------------------------------
    // Args:
    //   - beta: regression parameters (length p)
    //   - y: response vector (length n)
    //   - X: matrix of the features (n*p)
    //   - mi: vector of the number of trials, always 1 in the logit framework (length n)
    // Returns:
    //   - grad_ll: the gradient vector of the negative log-lik of the data (length p)
	// -------------------------------------------------------------------------------
	//Initialize variables
	unsigned int N = X.n_rows;
	unsigned int P = X.n_cols;
	vec wi = zeros(N);
	vec grad_ll = zeros(P);
	
	// Compute the gradient of the log-likelihood
	wi = comp_wi(X, beta);
	grad_ll = - X.t() * (y - mi % wi);
	return grad_ll;
}

// [[Rcpp::export]]
double optimal_step(vec beta, vec dir, vec y, mat X, vec mi, double c = 0.01, double alpha_max = 2.0, double rho = 0.5){
    // Computes the line search
    // ------------------------
    // Args:
    //   - beta: actual beta parameters we are moving from
    //   - dir: chosen descent direction
    //   - y: response vector (length n)
	//   - X: matrix of the features (n*p)
    //   - mi: vector of the number of trials, always 1 in the logit framework (length n)
    //   - c:
    //   - alpha_max: initial step size (maximum accepted value)
    //   - rho: shrinkage factor
    // Returns:
    //   - alpha_opt: the optimal step size for the requested direction
    // ----------------------------------------------------------------
	//Initialize variables
	double alpha_opt = alpha_max;
	double ll_act = log_lik(beta, y, X, mi);
	vec gradll_act = grad_loglik(beta, y, X, mi);

	//Backtrack search
	while(log_lik(beta + alpha_opt * dir, y, X, mi) > ll_act + c * alpha_opt * dot(gradll_act, dir)){
		alpha_opt = alpha_opt * rho;
	}

	return alpha_opt;
}

// [[Rcpp::export]]
mat GD_line_search(vec y, mat X, vec mi, vec beta0, unsigned int maxiter, double tol){
	// Function for the gradient descent of the logit model coupled with backtracking
	// ------------------------------------------------------------------------------
	// Args:
	//   - y: response vector (length n)
	//   - X: matrix of the features (n*p)
	//   - mi: vector of the number of trials, always 1 in the logit framework (length n)
	//   - beta0: initial regression parameters (length p)
	//   - maxiter: number of maximum iterations the algorithm will perform if not converging
	//   - tol: tolerance threshold to convergence
	// Returns:
	//   - betas: final optimal regression parameters
    // -------------------------------------------------
	//Initialize variables
	unsigned int P = X.n_cols;
	mat betas = zeros(P, maxiter);
	betas.col(0) = vec(beta0);
	
	vec ll = zeros(maxiter);
	ll(0) = log_lik(betas.col(0), y, X, mi);
	vec gradient = zeros(P);
	vec dir = zeros(P);
	
	vec alpha = zeros(maxiter);

    // for loop
	for (unsigned int iter = 1; iter < maxiter; iter++){

		// Compute the gradient and the descent direction
		gradient = grad_loglik(betas.col(iter - 1), y, X, mi);
		dir = - gradient;

		// Compute the optimal step size
		alpha(iter - 1) = optimal_step(betas.col(iter - 1), dir, y, X, mi);

		// Update the parameters and compute log-likelihood
		betas.col(iter) = betas.col(iter - 1) + alpha(iter - 1) * dir;
		ll(iter) = log_lik(betas.col(iter), y, X, mi);
				
		// Convergence check
		if (abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) < tol){
			Rcpp::Rcout << "Algorithm has converged after " << iter + 1 << " iterations." << std::endl;
			return betas.cols(0, iter);
		}
		else if (iter == maxiter && abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) >= tol){
			Rcpp::Rcout << "WARNING: the algorithm has not converged." << std::endl;
			break;
		}
	}
	return betas;
}

