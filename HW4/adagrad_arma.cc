#define ARMA_64BIT_WORD
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
	wi = 1.0 / (1.0 + exp(-sum(X % beta)));
	//Alternative:
	//wi = 1.0 / (1.0 + exp(- X.t() * beta));
	
	return wi;
}

// [[Rcpp::export]]
double log_lik(vec beta, vec y, mat X, vec mi, double lambda = 0.0){
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
	unsigned int N = y.n_elem;
	double ll = 0.0;
	vec wi = zeros(N);

	// Compute the log-likelihood
	wi = comp_wi(X, beta);
	ll = - sum(y % log(wi + 1E-10) + (mi - y) % log(1.0 - wi + 1E-10)) + lambda * as_scalar(sum(pow(beta, 2)));

	return ll;
}

// [[Rcpp::export]]
vec grad_loglik(vec beta, vec y, mat& X, vec mi, double lambda = 0.0){
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
	unsigned int N = y.n_elem;
	unsigned int P = beta.n_elem;
	vec wi = zeros(N);
	vec grad_ll = zeros(P);

	// Compute the gradient of the log-likelihood
	wi = comp_wi(X, beta);
	grad_ll = - X * (y - mi % wi) + 2.0 * lambda * beta;

	return grad_ll;
}


// [[Rcpp::export]]
List Ada_Grad(vec y, mat& X, vec mi, vec beta0, double eta = 1.0, double lambda = 0.0, unsigned int maxiter = 1000000, double tol = 1E-12){
	// Function for the Adaptive Gradient descent of the logit model
	// -------------------------------------------------------------
	// Args:
	//   - y: response vector (length n)
	//   - X: matrix of the features (n*p)
	//   - mi: vector of the number of trials, always 1 in the logit framework (length n)
	//   - beta0: initial regression parameters (length p)
	//   - maxiter: number of maximum iterations the algorithm will perform if not converging
	//   - tol: tolerance threshold to convergence
	//   - eta: correction for the step size
	// Returns:
	//   - betas: final optimal regression parameters
	//   - av_ll: average log-likelihood
    // ----------------------------------------------
	//Initialize variables
	unsigned int P = X.n_rows;
	unsigned int N = X.n_cols;
	//Initialize variables for the sampled values (explicit casting to matrix and vector, otherwise the functions will not work)
	vec y_samp = zeros(1);
	y_samp = y(0);
	mat X_samp = X.col(0);
	vec mi_samp = zeros(1);
	mi_samp = mi(0);
	//Initialize variables for the results (betas and log-likelihoods)
	mat betas = zeros(P, maxiter);
	betas.col(0) = vec(beta0);
	vec av_ll = zeros(maxiter);
	av_ll(0) = log_lik(betas.col(0), y_samp, X_samp, mi_samp, lambda);
	//Initialize other support variables
	unsigned int idx;
	vec hist_grad = zeros(P);
	vec grad = zeros(P);
	vec dir = zeros(P);

	vec alpha = zeros(maxiter);

    // for loop
	for (unsigned int iter = 1; iter < maxiter; iter++){
		// We select one data point
		idx = iter % N;
		//Rcpp::Rcout << "Data point sampled: " << idx << std::endl;
		y_samp = y(idx);
		X_samp = X.col(idx);
		//Rcpp::Rcout << "X row accessed" << std::endl;
		mi_samp = mi(idx);
		// Compute the gradient and the descent direction
		grad = grad_loglik(betas.col(iter - 1), y_samp, X_samp, mi_samp, lambda);

		hist_grad += pow(grad, 2);
		dir = - grad / (sqrt(hist_grad) + 1E-10);

		// Update the parameters and compute log-likelihood
		betas.col(iter) = betas.col(iter - 1) + eta * dir;

		av_ll(iter) = (av_ll(iter - 1) * iter + log_lik(betas.col(iter), y_samp, X_samp, mi_samp, lambda)) / (iter + 1);
		//Rcpp::Rcout << "Iteration " << iter << "." << std::endl;
		// // Convergence check
		// if (abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) < tol){
		// 	Rcpp::Rcout << "Algorithm has converged after " << iter + 1 << " iterations." << std::endl;
		// 	return betas.cols(0, iter);
		// }
		// else if (iter == maxiter && abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) >= tol){
		// 	Rcpp::Rcout << "WARNING: the algorithm has not converged." << std::endl;
		// 	break;
		// }
	}
	return Rcpp::List::create(
		_["betas"] = betas,
		_["loglik"] = av_ll
	);
}






// // [[Rcpp::export]]
// double optimal_step(vec beta, vec dir, vec y, mat X, vec mi, double c = 0.01, double alpha_max = 2.0, double rho = 0.5){
//     // Computes the line search
//     // ------------------------
//     // Args:
//     //   - beta: actual beta parameters we are moving from
//     //   - dir: chosen descent direction
//     //   - y: response vector (length n)
// 	//   - X: matrix of the features (n*p)
//     //   - mi: vector of the number of trials, always 1 in the logit framework (length n)
//     //   - c:
//     //   - alpha_max: initial step size (maximum accepted value)
//     //   - rho: shrinkage factor
//     // Returns:
//     //   - alpha_opt: the optimal step size for the requested direction
//     // ----------------------------------------------------------------
// 	//Initialize variables
// 	double alpha_opt = alpha_max;
// 	double ll_act = log_lik(beta, y, X, mi);
// 	vec gradll_act = grad_loglik(beta, y, X, mi);
//
// 	//Backtrack search
// 	while(log_lik(beta + alpha_opt * dir, y, X, mi) > ll_act + c * alpha_opt * dot(gradll_act, dir)){
// 		alpha_opt = alpha_opt * rho;
// 	}
//
// 	return alpha_opt;
// }


// // [[Rcpp::export]]
// mat GD_line_search(vec y, mat X, vec mi, vec beta0, unsigned int maxiter, double tol){
// 	// Function for the gradient descent of the logit model coupled with backtracking
// 	// ------------------------------------------------------------------------------
// 	// Args:
// 	//   - y: response vector (length n)
// 	//   - X: matrix of the features (n*p)
// 	//   - mi: vector of the number of trials, always 1 in the logit framework (length n)
// 	//   - beta0: initial regression parameters (length p)
// 	//   - maxiter: number of maximum iterations the algorithm will perform if not converging
// 	//   - tol: tolerance threshold to convergence
// 	// Returns:
// 	//   - betas: final optimal regression parameters
//     // -------------------------------------------------
// 	//Initialize variables
// 	unsigned int P = X.n_cols;
// 	mat betas = zeros(P, maxiter);
// 	betas.col(0) = vec(beta0);
//
// 	vec ll = zeros(maxiter);
// 	ll(0) = log_lik(betas.col(0), y, X, mi);
// 	vec gradient = zeros(P);
// 	vec dir = zeros(P);
//
// 	vec alpha = zeros(maxiter);
//
//     // for loop
// 	for (unsigned int iter = 1; iter < maxiter; iter++){
//
// 		// Compute the gradient and the descent direction
// 		gradient = grad_loglik(betas.col(iter - 1), y, X, mi);
// 		dir = - gradient;
//
// 		// Compute the optimal step size
// 		alpha(iter - 1) = optimal_step(betas.col(iter - 1), dir, y, X, mi);
//
// 		// Update the parameters and compute log-likelihood
// 		betas.col(iter) = betas.col(iter - 1) + alpha(iter - 1) * dir;
// 		ll(iter) = log_lik(betas.col(iter), y, X, mi);
//
// 		// Convergence check
// 		if (abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) < tol){
// 			Rcpp::Rcout << "Algorithm has converged after " << iter + 1 << " iterations." << std::endl;
// 			return betas.cols(0, iter);
// 		}
// 		else if (iter == maxiter && abs(ll(iter - 1) - ll(iter)) / abs(ll(iter - 1) + 1E-3) >= tol){
// 			Rcpp::Rcout << "WARNING: the algorithm has not converged." << std::endl;
// 			break;
// 		}
// 	}
// 	return betas;
// }




