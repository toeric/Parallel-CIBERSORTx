#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
using Matrix = arma::Mat<double>;

void split(const char*, vector<double>&);
Matrix read_bulk(string);
vector<string> read_batch(string);
inline double string_to_double(string);


Matrix read_bulk(string file_name) {

	vector<vector<double>> v;
	vector<string> batch;

	ifstream ifs(file_name, ios::in);
	if (!ifs.is_open()) {
		cout << "Fail to open bulk data!\n";
	}else {
		string s;
		getline(ifs, s);
		while (getline(ifs, s)) {
			vector <double> tmp;
			split(s.c_str(), tmp);
			v.push_back(tmp);
		}	
	}
	Matrix M(v.size(), v[0].size());

	for (int i = 0; i < v.size(); ++i) {
		for (int j = 0; j < v[0].size(); ++j) {
			M(i, j) = v[i][j];
		}
	}
	return M;
}

vector<string> read_gene(string file_name) {

	vector<string> re;
	char c = '\t';
	ifstream ifs(file_name, ios::in);
	if (!ifs.is_open()) {
		cout << "Fail to open read gene data!\n";
	}else {
		string s;
		getline(ifs, s);
		while (getline(ifs, s)) {
			const char *begin = s.c_str();
			const char* str = s.c_str();
			while (*str != c && *str)
				str++;
			re.push_back(string(begin, str));
		}	
	}
	return re;
}

vector<string> read_batch(string file_name) {
	vector<string> b;
	ifstream ifs(file_name, ios::in);
	string s;
	if (!ifs.is_open()) {
		cout << "Fail to open batch data!\n";
	}else {
		ifs >> s;
		while(ifs >> s){
			ifs >> s;
			b.push_back(s);
		}
	}
	return b;
}

void split(const char *str, vector<double>& result) {
	char c = '\t';
	bool first = true;
	do {
		const char *begin = str;
		while (*str != c && *str)
			str++;
		if (first)
			first = false;
		else 
			result.push_back(string_to_double(string(begin, str)));
	} while (0 != *str++);
}

inline double string_to_double(string s) {
	double result = 0;
	int weight = -1;
	bool dot = false; 
	for (int i = 0; i < s.size(); ++i) {	
		if (s[i] == '.') {dot = true;}
		else if (!dot) result = 10 * result + (s[i] - '0');
		else { result += pow(10, weight) * (s[i] - '0'); weight -= 1;}
	}
	return result;
}
						
#endif
