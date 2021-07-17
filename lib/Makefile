EXENAME=../test

CC = g++ -std=c++11
CFLAGS = -Wall -Wconversion -O3 -fPIC 
SHVER = 2
OS = $(shell uname)
ARMA_POS = /home/blah/armadillo-10.5.1/include
ARMA_FL = -I $(ARMA_POS) -DARMA_DONT_USE_WRAPPER 

all: lib svm.o main 

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CC) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm.o: svm.cpp svm.h
	$(CC) $(CFLAGS) -c svm.cpp

train: train.cpp svm.o
	$(CC) $(CFLAGS) train.cpp svm.o -o train -lm

main: ../main.cpp svm.o
	$(CC) ../main.cpp svm.o -o $(EXENAME) -O2 $(ARMA_FL) -lblas -llapack -lpthread

.PHONY: clean
clean:
	rm -f $(EXENAME)
	rm -f *~ svm.o libsvm.so.$(SHVER)
#	rm *.txt

