#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <map>

using namespace std;
using Matrix = arma::Mat<double>;

Matrix model_matrix(vector<string>& batch) {
	vector<string> unique_vector;
	for (int i = 0; i < batch.size() ; ++i) {
		if (find(unique_vector.begin(), unique_vector.end(), batch[i]) == unique_vector.end())
			unique_vector.push_back(batch[i]);
	}	
	sort(unique_vector.begin(), unique_vector.end());
	Matrix M(batch.size(), unique_vector.size());
	for (int i = 0; i < batch.size() ; ++i) {
		auto it = find(unique_vector.begin(), unique_vector.end(), batch[i]);
		int idx = distance(unique_vector.begin(), it);
		M(i, idx) = 1;
	}
	return M;
}


int nlevels(vector<string> v) {
	sort(v.begin(), v.end());
	int unique_count = unique(v.begin(), v.end()) - v.begin();
	return unique_count;
}

map<string, vector<int>> build_batches(vector<string> batch) {
	map<string, vector<int>> batches;
	for(int i = 0; i < batch.size(); ++i) {
		auto iter = batches.find(batch[i]);
		if (iter == batches.end()) {
			vector<int> tmp{i};
			batches[batch[i]] = tmp;
		}else {
			batches[batch[i]].push_back(i);
		}
	}
	return batches;
}

arma::Row<int> build_n_batches(map<string, vector<int>> batches) {
	vector<string> keys;
	for(auto it = batches.begin(); it!= batches.end(); it++) {
		keys.push_back(it->first);
	}
	arma::Row<int> n_batches(keys.size());
	for (int i = 0; i < keys.size(); ++i) {
		n_batches[i] = batches[keys[i]].size();
	}
	return n_batches;
}

Matrix rowVars(Matrix& m, vector<int> idx = {-1}) {
	if (idx[0] == -1)
		return arma::var(m, 0, 1);
	else {
		Matrix r_m(m.n_rows, idx.size());
		for (int i = 0; i < idx.size(); ++i) {
			for (int j = 0; j < m.n_rows; ++j) {
				r_m(j, i) = m(j, idx[i]);
			}
		}
		return arma::var(r_m, 0, 1);
	}
}

Matrix aprior(Matrix& m) {
	Matrix r_m(m.n_rows, 1);
	Matrix _mean(m.n_rows, 1), _s2(m.n_rows, 1);
	_mean = arma::mean(m, 1);
	_s2 = arma::var(m, 0, 1);
	for (int i = 0; i < m.n_rows; ++i) {
		r_m[i] = ((2 * _s2[i]) + pow(_mean[i], 2)) / _s2[i]; 
	}
	return r_m.t();
}

Matrix bprior(Matrix& m) {
	Matrix r_m(m.n_rows, 1);
	Matrix _mean(m.n_rows, 1), _s2(m.n_rows, 1);
	_mean = arma::mean(m, 1);
	_s2 = arma::var(m, 0, 1);
	for (int i = 0; i < m.n_rows; ++i) {
		r_m[i] = ((_mean[i] * _s2[i]) + pow(_mean[i], 3)) / _s2[i]; 
	}
	return r_m.t();
}

Matrix mul_each(Matrix a, Matrix b) {
	Matrix c(1, a.n_cols);
	for (int i = 0; i < a.n_cols; ++i) c[i] = a[i] * b[i];
	return c;
}

Matrix postmean(Matrix g_hat, double g_bar, Matrix n, Matrix d_star, double t2) {
	return (t2 * mul_each(n, g_hat) + d_star * g_bar) / (t2 * n + d_star);
}

Matrix postvar(Matrix sum2, Matrix n, double a, double b) {
	return (0.5 * sum2 + b).t() / (n / 2 + a - 1);
}

double cal_change(Matrix& g_o, Matrix& g_n, Matrix& d_o, Matrix& d_n) {
	Matrix g(1, g_o.n_cols), d(1, d_o.n_cols);
	for (int i = 0; i < g_o.n_cols; ++i) g[i] = (abs(g_o[i] - g_n[i])) / g_o[i];
	for (int i = 0; i < d_o.n_cols; ++i) d[i] = (abs(d_o[i] - d_n[i])) / d_o[i];
	if (g.max() > d.max()) return g.max();
	else return d.max();

}

Matrix it_sol(Matrix s_data, vector<int> idx, Matrix g_hat, Matrix d_hat, double g_bar, double t2, double a, double b, double conv = 0.0001) {
	Matrix tmp_s_data(s_data.n_rows, idx.size());
	for (int i = 0; i < s_data.n_rows; ++i) 
		for (int j = 0; j < idx.size(); ++j)
			tmp_s_data(i, j) = s_data(i, idx[j]);

	Matrix n(1, s_data.n_rows);
	for (int i = 0; i < s_data.n_rows; ++i) n(i) = idx.size();
	Matrix g_old = g_hat;
	Matrix d_old = d_hat;
	Matrix g_new, d_new;
	double change = 1.0;
	int count = 0;
	while(change > conv) {
		
		g_new = postmean(g_hat, g_bar, n, d_old, t2);
		arma::Row<int> one_s_data_col(idx.size(), arma::fill::ones);
		Matrix tmp = tmp_s_data - g_new.t() * one_s_data_col;
		Matrix sum2 = arma::sum(tmp.transform([](double val){ return pow(val, 2);}), 1);
		d_new = postvar(sum2, n, a, b);
		
		change = cal_change(g_old, g_new, d_old, d_new);
	
		g_old = g_new;
		d_old = d_new;
		count += 1;
	}
	
	Matrix re;
	re.insert_rows(0, g_new);
	re.insert_rows(1, d_new);

	return re;
}


#endif
