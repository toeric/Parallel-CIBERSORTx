# Parallel-CIBERSORT

## Setting up Environment

## Install BLAS and LAPACK
```
apt-get install libblas-dev liblapack-dev
```

### Install armadillo from source
```
wget http://sourceforge.net/projects/arma/files/armadillo-10.5.1.tar.xz
tar Jxvf armadillo-10.5.1.tar.xz  
cd armadillo-10.5.1

cmake
make
make install
```

### Install Parallel-CIBERSORTx
```
git clone https://github.com/toeric/Parallel-CIBERSORTx.git
cd lib
make
```

### Run Parallel-CIBERSORTx
```cpp
//main.cpp

#include <armadillo>
#include <string>
#include "lib/CIBERSORTx.h"

int main(int arcg, char *argv[]) {

	std::string M = argv[1]; // Bulk data file name
	std::string S = argv[2]; // Signature Matrix file name

	// false: deconvolution without batch correction
	// true: process B-mode correction
	bool batch = false; 
	
	//Number of thread 
	int thread_num = 48;

	CIBERSORTx model(M, S); 
	model.dodecomposition(batch, thread_num);

	 return 0;
}
```

Run the specific data:
```
./test data/Mix_10.txt data/Sig.txt
```

Actually, we had run bulk data with 10, 600 and 10000 samples to test our pakage.
Since the bulk data with 600 and 10000 samples is too big,
we just upload bulk data with 10 samples (Mix\_10.txt) to github. 


