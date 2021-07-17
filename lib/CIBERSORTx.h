#include <string>
#include <vector>
#include <sstream>

#include "SVR.h"
#include "ComBat.h"

class CIBERSORTx {
	public:
		CIBERSORTx (std::string M, std::string S) {
			Mix_file = M;
			Sig_file = S;
		};
		~CIBERSORTx (){};

		void dodecomposition (bool batch, int thread_num) {
				
				string final_result_file = "CIBERSORTx_result.txt";
				std::vector<std::future<std::tuple<vector<pair<int, vector<double>>>, vector<pair<int, double>>>>> futures;
				for (int i = 0; i < 3; i++){
					if (!batch){ 
						auto fut = std::async(std::launch::async, SVR, this->Sig_file, this->Mix_file, (i + 1)*0.25, std::to_string((i + 1) * 25), false, true, thread_num);
						futures.push_back(std::move(fut));
					} else{
						auto fut = std::async(std::launch::async, SVR, this->Sig_file, this->Mix_file, (i + 1)*0.25, std::to_string((i + 1) * 25), true, true, thread_num);
						futures.push_back(std::move(fut));
					}
				}
				vector<vector<pair<int, vector<double>>>> weight;
				vector<vector<pair<int, double>>> rmse;
				for (auto& fut : futures) {    //Iterate through in the order the future was created
					auto vec = fut.get();    //Get the result of the future
					weight.emplace_back(get<0>(vec));
					rmse.emplace_back(get<1>(vec));
				}
				vector<pair<int, vector<double>>> final_weight;
				for (int i = 0; i < rmse[0].size(); ++i)
				{
						double min = std::min({ rmse[0][i].second, rmse[1][i].second, rmse[2][i].second });
						if (min == rmse[0][i].second)
							final_weight.push_back(weight[0][i]);
						if (min == rmse[1][i].second)
							final_weight.push_back(weight[1][i]);
						if (min == rmse[2][i].second)
							final_weight.push_back(weight[2][i]);					
				}
				ofstream final_weight_result;
				final_weight_result.open(final_result_file, std::ios_base::app);
				for (size_t i = 0; i < final_weight.size(); ++i)
					for (auto p : final_weight){
						if (p.first == i){
							for (auto e : p.second){
								final_weight_result << e << '\t';
							}
							final_weight_result << '\n';
						}
					}

				final_weight_result.close();
			
			if (batch) {
				std::ifstream result_file;
				result_file.open("CIBERSORTx_result.txt");
				std::string s;
				double tmp_val;
				std::vector<std::vector<double>> ma_cell_type;
				while(getline(result_file, s)) {
					stringstream ss(s);
					std::vector<double> each_row;
					while(ss >> tmp_val) {
						each_row.push_back(tmp_val);
					}
					ma_cell_type.push_back(each_row);
				}
				Matrix first_result(ma_cell_type.size(), ma_cell_type[0].size());
				for (int i = 0; i < ma_cell_type.size(); ++i) {
					for (int j = 0; j < ma_cell_type[0].size(); ++j) {
						first_result(i, j) = ma_cell_type[i][j];
					}
				}

				std::ifstream used_bulk_file;
				used_bulk_file.open("used_bulk.txt");
				std::vector<std::vector<double>> ma_used_bulk;
				while(getline(used_bulk_file, s)) {
					stringstream ss(s);
					std::vector<double> each_row;
					while(ss >> tmp_val) {
						each_row.push_back(tmp_val);
					}
					ma_used_bulk.push_back(each_row);
				}
				Matrix used_bulk(ma_used_bulk.size(), ma_used_bulk[0].size());
				for (int i = 0; i < ma_used_bulk.size(); ++i) {
					for (int j = 0; j < ma_used_bulk[0].size(); ++j) {
						used_bulk(i, j) = ma_used_bulk[i][j];
					}
				}

				std::ifstream used_sig_file;
				used_sig_file.open("used_sig.txt");
				std::vector<std::vector<double>> ma_used_sig;
				while(getline(used_sig_file, s)) {
					stringstream ss(s);
					std::vector<double> each_row;
					while(ss >> tmp_val) {
						each_row.push_back(tmp_val);
					}
					ma_used_sig.push_back(each_row);
				}
				Matrix used_sig(ma_used_sig.size(), ma_used_sig[0].size());
				for (int i = 0; i < ma_used_sig.size(); ++i) {
					for (int j = 0; j < ma_used_sig[0].size(); ++j) {
						used_sig(i, j) = ma_used_sig[i][j];
					}
				}

				Matrix new_bulk = used_sig * first_result.t();
			
				std::vector<std::string> batch;
				for (int i = 0; i < new_bulk.n_cols; ++i) {
					batch.insert(batch.begin(), "1");
					batch.push_back("2");
				}

				Matrix old_and_new = arma::join_rows(used_bulk, new_bulk);
				Matrix Combat_result = ComBat(old_and_new, batch);				
				Matrix M_star(Combat_result.n_rows, used_bulk.n_cols);
				for (int i = 0; i < used_bulk.n_cols; ++i) {
					M_star.col(i) = Combat_result.col(i);
				}

				ofstream M_star_file;
				M_star_file.open("M_star.txt");
				for (int i = 0; i <= M_star.n_cols; ++i) {
					M_star_file << "each_cluster" << '\t';
				}
				M_star_file << '\n';
				for (int i = 0; i < M_star.n_rows; ++i) {
					M_star_file << "each_gene" << '\t';
					for (int j = 0; j < M_star.n_cols - 1; ++j) {
						M_star_file << M_star(i, j) << '\t';
					}
					M_star_file << M_star(i, M_star.n_cols - 1) << '\n';
				}
				M_star_file.close();
				
				this->Mix_file = "M_star.txt";
				this->Sig_file = "M_star_sig.txt";
				string final_result_file = "CIBERSORTx_result_after_Bmode_correction.txt";
				std::vector<std::future<std::tuple<vector<pair<int, vector<double>>>, vector<pair<int, double>>>>> futures;
				for (int i = 0; i < 3; i++){
					auto fut = std::async(std::launch::async, SVR, this->Sig_file, this->Mix_file, (i + 1)*0.25, std::to_string((i + 1) * 25), true, false, thread_num);
					futures.push_back(std::move(fut));
				}
				vector<vector<pair<int, vector<double>>>> weight;
				vector<vector<pair<int, double>>> rmse;
				for (auto& fut : futures) {    //Iterate through in the order the future was created
					auto vec = fut.get();    //Get the result of the future
					weight.emplace_back(get<0>(vec));
					rmse.emplace_back(get<1>(vec));
				}
				vector<pair<int, vector<double>>> final_weight;
				for (int i = 0; i < rmse[0].size(); ++i){
						double min = std::min({ rmse[0][i].second, rmse[1][i].second, rmse[2][i].second });
						if (min == rmse[0][i].second)
							final_weight.push_back(weight[0][i]);
						if (min == rmse[1][i].second)
							final_weight.push_back(weight[1][i]);
						if (min == rmse[2][i].second)
							final_weight.push_back(weight[2][i]);					
				}
				ofstream final_weight_result;
				final_weight_result.open(final_result_file, std::ios_base::app);
				for (size_t i = 0; i < final_weight.size(); ++i)
					for (auto p : final_weight){
						if (p.first == i){
							for (auto e : p.second)
							{
								final_weight_result << e << '\t';
							}

							final_weight_result << '\n';
						}
					}
				final_weight_result.close();
			}
		};

	private:
		std::string Mix_file;
		std::string Sig_file;
	
};
