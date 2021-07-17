#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include <iostream>
#include <utility>
#include <vector>
#include <armadillo>
#include <mutex>

static char *line = NULL;
static int max_line_len;

class Train
{
public:
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;
	struct svm_node *x_space;
	int cross_validation;
	int nr_fold;

	std::vector<double> sv_coefs;
	std::vector<std::vector<double>> SVs;

	void get_result(std::vector<std::pair<int, std::vector<double>>> &re, int b)
	{
		std::vector<double> v;
		arma::colvec alpha(sv_coefs);
		arma::Mat<double> X(sv_coefs.size(), SVs[0].size());
		for (int i=0; i<sv_coefs.size(); ++i)
			for (int j=0; j<SVs[0].size(); ++j)
				X(i, j) = SVs[i][j];

		arma::rowvec W = alpha.t() * X;
		for (size_t i = 0; i < W.n_elem; ++i) 
			if (W(i) < 0) W(i) = 0;
		
		int sum_W = arma::sum(W);
		for (size_t i = 0; i < W.n_elem; ++i) {
			if (sum_W == 0) W(i) = 0;
			else W(i) = W(i)/sum_W;
			v.emplace_back(W(i));
		}

		re[b] = std::make_pair(b, v);
	}

	void model_vectorize()
	{
		const svm_parameter& param = model->param;
		int nr_class = model->nr_class;
		int l = model->l;
		const double * const *sv_coef = model->sv_coef;
		const svm_node * const *SV = model->SV;
		SVs = std::vector<std::vector<double>> (l, std::vector<double> ());

		for(int i=0;i<l;i++)
		{
			for(int j=0;j<nr_class-1;j++)
				sv_coefs.emplace_back(sv_coef[j][i]);

			const svm_node *p = SV[i];

			if(param.kernel_type == PRECOMPUTED)
				std::cout << "precomputed: " << (int)(p->value) << '\n';
			else
				while(p->index != -1)
					SVs[i].emplace_back(p->value), p++;
		}
	}

	void get_model(const double nu) 
	{
		char nu_buf[256];
		sprintf(nu_buf, "%f", nu);

		int size = 9;
		const char* cmd[] = {
			"./shit", "-s", "4", "-n", nu_buf, "-t", "0", "-h", "0"
		};
		const char** cmd_ptr = cmd;

		parse_command_line(size, cmd_ptr);

		const char* error_msg; 
		error_msg = svm_check_parameter(&prob, &param);
		if(error_msg)
		{
			fprintf(stderr,"ERROR: %s\n",error_msg);
			exit(1);
		}

		if(cross_validation)
			do_cross_validation();
		else 
		{
			model = svm_train(&prob, &param);
			model_vectorize();
			svm_free_and_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		free(line);
	}

	void exit_with_help()
	{
		printf(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		);
		exit(1);
	}

	void exit_input_error(int line_num)
	{
		fprintf(stderr,"Wrong input format at line %d\n", line_num);
		exit(1);
	}

	static char* readline(FILE *input)
	{
		int len;

		if(fgets(line,max_line_len,input) == NULL)
			return NULL;

		while(strrchr(line,'\n') == NULL)
		{
			max_line_len *= 2;
			line = (char *) realloc(line,max_line_len);
			len = (int) strlen(line);
			if(fgets(line+len,max_line_len-len,input) == NULL)
				break;
		}
		return line;
	}

	void parse_command_line(int argc, const char **argv);
	void do_cross_validation();
};

void print_null(const char *s) {}
void Train::parse_command_line(int argc, const char **argv)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);
}

void Train::do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}
