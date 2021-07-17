#include <string>
#include <armadillo>
#include <iostream>
#include <fstream>

#include <chrono>
#include <thread>
#include <vector>
#include <string>
//#include <unistd.h>

#include "lib/CIBERSORTx.h"

int main(int arcg, char *argv[]) {
	

	std::string M = argv[1];
	std::string S = argv[2];

	bool batch = false;
	int thread_num = 48;

	CIBERSORTx model(M, S);

	model.dodecomposition(batch, thread_num);

	return 0;
}

