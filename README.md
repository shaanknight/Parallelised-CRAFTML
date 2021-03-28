# Parallelised Implementation of CRAFTML Algorithm
##### CRAFTML : Efficient Clustering-based Random Forest for Extreme Multi-label Learning 
#####  A hybrid MPI + OpenMP implementation built on C++, reduced to five cores 
###### Algorithm : 
The entire algorithm for training is well described in the paper : http://proceedings.mlr.press/v80/siblini18a/siblini18a.pdf .
The testing and the details of parallelisation is well covered in CRAFTML.pdf file .
###### Benchmarking : 
With the available resources we have checked the performance of our code against 5 popular datasets available at http://manikvarma.org/downloads/XC/XMLRepository.html, namely, Mediamill, Bibtex, Delicious, EURLex-4K and Wiki10-31K.
###### Compilation and Execution : 
The .h files are additional files required for running of any of the codes for generating hash matrices required for projection .
Below mentioned is the format and details of 3 codes included in the folder :
- CRAFTML_openmp.cpp : A single tree building and testing code, optimized by OpenMP threads .

Formatted to work for large datasets.

Compile : 
```sh
g++ -fopenmp CRAFTML_openmp.cpp
```
Run : 
```sh
./a.out train_filename test_filename
```

e.g : 
```
./a.out Eurlex/eurlex_train.txt Eurlex/eurlex_test.txt
```

- CRAFTML_parallel_large.cpp : A multiple tree building and testing code, parallelized by MPI and and further
optimized by OpenMP . 

Formatted to work for large datasets.

Compile : 
```sh
mpiCC -fopenmp CRAFTML_parallel_large.cpp -std=c++11
```

Run : 
```sh
mpirun -np num_procs ./a.out train_filename test_filename num_trees
```

e.g : 
```sh
mpirun -np 4 ./a.out Eurlex/eurlex_train.txt Eurlex/eurlex_test.txt 5 
```
desired to run 4 processes, each implementing 5 trees

- CRAFTML_parallel_small.cpp : A multiple tree building and testing code, parallelized by MPI and and further

optimized by OpenMP . Formatted to work for small datasets.

Compile : 
```sh
mpiCC -fopenmp CRAFTML_parallel_small.cpp -std=c++11
```
Run : 
```sh
mpirun -np num_procs ./a.out data_filename train_split test_split num_trees
```
e.g :  
```sh
mpirun -np 5 ./a.out Bibtex/Bibtex_data.txt Bibtex/bibtex_trSplit.txt Bibtex/bibtex_tstSplit.txt 10
```
desired to run 5 processes, each implementing 10 trees













   
 
