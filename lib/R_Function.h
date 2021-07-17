#include <armadillo>
#include <string>
#include <math.h>

using Matrix  = arma::Mat<double>;

Matrix sweep(Matrix X, arma::rowvec w){
	Matrix X_(X.n_rows, X.n_cols);
	for(auto i=0; i<X.n_rows; i++)
		X_.row(i) = X.row(i) % w;
	return X_;
}

arma::colvec apply(Matrix u){
	arma::colvec k;
	k = sum(u,1);
	return k;
}

double rmse_cal(arma::colvec k, arma::colvec y){
	double err;
	err = std::pow(arma::mean(arma::square(k-y)), 0.5);
	
	return err;
}