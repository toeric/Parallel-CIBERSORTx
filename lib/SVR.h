#include <string>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <tuple>

#include "parser.h"
#include "ThreadPool.h"
#include "svm.h"
#include "train.hpp"
#include "R_Function.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using Matrix  = arma::Mat<double>;
using namespace std;

std::tuple<vector<pair<int, vector<double>>>, vector<pair<int, double>>> SVR (string sig_file, string bulk_file, const double nu, string ID, bool batch, bool need_gene_name, int thread_num) {
	
	//bool batch = false;
	Matrix sig, bulk;
	vector<string> sig_gene, bulk_gene;
	vector<int> sig_gene_idx, bulk_gene_idx;
//	string result_file = "D:/Course/Parallel/Cibersortx/Cibersortx_Batch/V2/V2/SVR_"+ID+"_result.txt";

	sig = read_bulk(sig_file);
	bulk = read_bulk(bulk_file);
	sig_gene = read_gene(sig_file);
	bulk_gene = read_gene(bulk_file);

	if (need_gene_name) {
		for (int i = 0; i < sig_gene.size(); ++i) {
			for (int j = 0; j < bulk_gene.size(); ++j) {
				if (sig_gene[i] == bulk_gene[j]) {
					sig_gene_idx.push_back(i);
					bulk_gene_idx.push_back(j);
					break;
				}
			}
		}
	}else {
		for (int i = 0; i < sig_gene.size(); ++i) {
			sig_gene_idx.push_back(i);
			bulk_gene_idx.push_back(i);
		}
	}
	
	Matrix sig_train(sig_gene_idx.size(), sig.n_cols);
	arma::rowvec sig_as_vec = sig.row(sig_gene_idx[0]);
	for (int i = 0; i < sig_gene_idx.size(); ++i) {
		sig_train.row(i) = sig.row(sig_gene_idx[i]);
		if (i > 0) {
			sig_as_vec = arma::join_rows(sig_as_vec, sig_train.row(i));
		}
	}

	if(need_gene_name) {
		double sig_train_ave = arma::mean(arma::mean(sig_train));
		double sig_train_std = arma::stddev(sig_as_vec);
		sig_train = (sig_train - sig_train_ave) / sig_train_std;
	}
	
	Matrix bulk_train(bulk_gene_idx.size(), bulk.n_cols);
	for (int i = 0; i < bulk_gene_idx.size(); ++i) {
		bulk_train.row(i) = bulk.row(bulk_gene_idx[i]);
	}

	if (need_gene_name) {
		for (int i = 0; i < bulk_train.n_cols; ++i) {
			double average = arma::mean(bulk_train.col(i));
			double std = arma::stddev(bulk_train.col(i));
			for (auto j = bulk_train.col(i).begin(); j != bulk_train.col(i).end(); ++j) {
				*j = (*j - average) / std;
			}
		}
	}


	if (nu == 0.25 && need_gene_name == true && batch) {
		ofstream used_bulk;
		used_bulk.open("used_bulk.txt");
		for (int i = 0; i < bulk_train.n_rows; ++i) {
			for (int j = 0; j < bulk_train.n_cols; ++j) {
				used_bulk << bulk_train(i, j) << '\t';
			}
			used_bulk << '\n';
		}
		used_bulk.close();


		ofstream used_sig;
		used_sig.open("used_sig.txt");
		for (int i = 0; i < sig_train.n_rows; ++i) {
			for (int j = 0; j < sig_train.n_cols; ++j) {
				used_sig << sig_train(i, j) << '\t';
			}
			used_sig << '\n';
		}
		used_sig.close();

		ofstream M_star_sig_file;
		M_star_sig_file.open("M_star_sig.txt");
		for (int i = 0; i <= sig_train.n_cols; ++i) {
			M_star_sig_file << "each_cluster" << '\t';
		}
		M_star_sig_file << '\n';
		for (int i = 0; i < sig_train.n_rows; ++i) {
			M_star_sig_file << "each_gene" << '\t';
			for (int j = 0; j < sig_train.n_cols - 1; ++j) {
				M_star_sig_file << sig_train(i, j) << '\t';
			}
			M_star_sig_file << sig_train(i, sig_train.n_cols - 1) << '\n';
		}
		M_star_sig_file.close();

	}

//	ofstream clean_result_file(result_file);
//	clean_result_file << "";
//	clean_result_file.close();

	ThreadPool pool(thread_num);
	vector<future<void>> future_re;

	vector<pair<int, double>> compare_vec;
	vector<pair<int, vector<double>>> re(bulk.n_cols);
	
	for (int b = 0; b < bulk.n_cols; ++b) {
		
		future_re.emplace_back(
			pool.enqueue([&re, &sig_train, &bulk_train, b, nu, &compare_vec]{
				Train* R = new Train();
				R->prob.l  = sig_train.n_rows;
				R->prob.y  = Malloc(double, R->prob.l);
				R->prob.x  = Malloc(struct svm_node*, R->prob.l);
				R->x_space = Malloc(struct svm_node , R->prob.l*(sig_train.n_cols+1));

				int addr = 0;
				for (int i = 0; i < sig_train.n_rows; ++i) {
					R->prob.y[i] = bulk_train(i, b);
					R->prob.x[i] = &R->x_space[addr];
					for (int j=0; j<sig_train.n_cols+1; ++j)
					{
						R->x_space[addr].index = (j == sig_train.n_cols) ? -1 : j+1;
						R->x_space[addr].value = (j == sig_train.n_cols) ?  0 : sig_train(i, j);
						++addr;
					}
				}

				R->get_model(nu);
				R->get_result(re, b);
			})
		);
	}
	
	for(auto &r:future_re)
		r.get();

	vector<pair<int, vector<double>>>final_weight;

	for (size_t i = 0; i < bulk.n_cols; ++i)
	{
		Matrix sweep_train(sig_train.n_rows, sig_train.n_cols);
		arma::colvec k(sig_train.n_rows);
		double rmse = 0.0;
		for (auto p : re)
		{
			if (p.first == i)
			{
				sweep_train = sweep(sig_train, p.second);
				k = apply(sweep_train);
				rmse = rmse_cal(k, bulk_train.col(i));
				compare_vec.push_back(std::make_pair(i, rmse));
				final_weight.emplace_back(std::make_pair(p.first, p.second));
			}
		}
	}
/*
	ofstream SVR_result;
	SVR_result.open(result_file, std::ios_base::app);
	for(size_t i=0; i<bulk.n_cols; ++i)
		for (auto p : re)
		{
			if (p.first == i)
			{
				for (auto e : p.second)
				{
					SVR_result << e << '\t';
				}
			
				SVR_result << '\n';
			}
		}
	
			
	SVR_result.close();
	*/
	std::tuple<vector<pair<int, vector<double>>>, vector<pair<int, double>>> weight_and_rmse = std::make_tuple(final_weight, compare_vec);
	return weight_and_rmse;
}
