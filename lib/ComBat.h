#ifndef COMBAT_H
#define COMBAT_H

#include <stdio.h>
#include <map>
#include <armadillo>
#include <math.h>
#include "parser.h"
#include "utils.h"

using Matrix = arma::Mat<double>;

Matrix ComBat(Matrix org_dat,vector<string> batch){

	int var_not_zero_count = 0;
	vector<int> not_zero_row;
	Matrix Var_matrix = var(org_dat, 0, 1);

	for (int i = 0; i < Var_matrix.n_rows; ++i) {
		if (Var_matrix(i, 0) != 0)
			var_not_zero_count += 1;
	}

	Matrix dat(var_not_zero_count, org_dat.n_cols);
	int dat_idx = 0;
	for (int i = 0; i < Var_matrix.n_rows; ++i) {
		if (Var_matrix(i, 0) != 0) {
			dat.row(dat_idx) =  org_dat.row(i);
			dat_idx ++;
			not_zero_row.push_back(i);
		}
	}

	cout << "Find " << var_not_zero_count << " rows with not 0 varaince" << endl;
	
	Matrix design = model_matrix(batch);
	
	int n_batch = nlevels(batch);
	map<string, vector<int>> batches = build_batches(batch);
	arma::Row<int> n_batches = build_n_batches(batches);
	vector<string> keys;
	for(auto it = batches.begin(); it!= batches.end(); it++) {
		keys.push_back(it->first);
	}
	int n_array = batch.size();

	Matrix B_hat = arma::solve(design.t() * design, design.t() * dat.t());
	
	arma::Row<double> norm_n_batches(n_batches.size());
	for (int i = 0; i < n_batches.size(); ++i) norm_n_batches(i) = n_batches(i);
	norm_n_batches.transform([=](double val){ return (val/n_array); });
	Matrix grand_mean = norm_n_batches * B_hat; 
	
	Matrix var_pooled = (dat - (design * B_hat).t());
	var_pooled.transform([](double val){return pow(val, 2);});
	arma::Col<double> tmp_average_n_array(n_array);
	tmp_average_n_array.fill(1.0 / n_array);
	var_pooled = var_pooled * tmp_average_n_array;

	arma::Row<double> tmp_one_n_array(n_array, arma::fill::ones);
	Matrix stand_mean = grand_mean.t() *  tmp_one_n_array;

	Matrix s_data_denominator = var_pooled;
	s_data_denominator.transform([](double val){ return pow(val, 0.5);});
	s_data_denominator = s_data_denominator * tmp_one_n_array;
	Matrix s_data = (dat - stand_mean) / s_data_denominator;

	Matrix batch_design = design;

	Matrix gamma_hat = arma::solve((batch_design.t() * batch_design), (batch_design.t() * s_data.t()));

	Matrix delta_hat;
	for (auto it = batches.begin(); it != batches.end(); ++it) {
		delta_hat.insert_cols(delta_hat.n_cols, rowVars(s_data, batches[it->first]));
	}
	delta_hat = delta_hat.t();

	Matrix gamma_bar = arma::mean(gamma_hat, 1).t(); 
	Matrix t2 = rowVars(gamma_hat).t();

	std::cout << "Finding parametric adjustment" << std::endl;

	Matrix a_prior = aprior(delta_hat);
	Matrix b_prior = bprior(delta_hat);
	
	Matrix gamma_star;
	Matrix delta_star;
	for (int i = 0; i < n_batch; ++i) {
		Matrix temp = it_sol(s_data, batches[keys[i]], gamma_hat.row(i), delta_hat.row(i), gamma_bar[i], t2[i], a_prior[i], b_prior[i]);
		gamma_star.insert_rows(gamma_star.n_rows, temp.row(0));
		delta_star.insert_rows(delta_star.n_rows, temp.row(1));
	}
	
	cout << "Adjusting the Data" << endl;

	Matrix bayesdata = s_data;

	int j = 0; 
	for (int it = 0; it < keys.size(); ++it) {
		Matrix mess_matrix = delta_star.row(j);
		mess_matrix .transform([](double val){ return pow(val, 0.5);});
		for (int i = 0; i < batches[keys[it]].size(); i++) {
			arma::Row<double> one_n_batches(n_batches[j], arma::fill::ones);
			bayesdata.col(batches[keys[it]][i]) = (bayesdata.col(batches[keys[it]][i]) - (batch_design.row(batches[keys[it]][i]) * gamma_star).t()) 
																								/ (mess_matrix.t() /** one_n_batches*/);
		}
		j += 1;
	}

	bayesdata = bayesdata % s_data_denominator + stand_mean;
	int bayes_idx = 0;
	Matrix final_bayesdata(org_dat.n_rows, org_dat.n_cols);
	for (int i = 0; i < org_dat.n_rows; ++i) {
		if (find(not_zero_row.begin(), not_zero_row.end(), i) != not_zero_row.end()) {
			final_bayesdata.row(i) =  bayesdata.row(bayes_idx);
			bayes_idx ++;
		} else {
			final_bayesdata.row(i) = org_dat.row(i);
		}
	}
	return final_bayesdata;
}

#endif 
