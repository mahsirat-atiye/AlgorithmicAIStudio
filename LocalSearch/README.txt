This is a python 3.6 code to implement simulated annealing & hill climbing algorithm to solve following problem:
given A and b for AX = b, where num of columns of A are more than num of rows,
the goal is to find a vector of X variables  which best satisfies AX=b

this code will find a vector of X' variables through simulated annealing & hill climbing algorithms by
objective function: MIN sigma (AX' - b) ^ 2

INPUTS:
1. address of txt file in response to following: "Please enter address of txt file like: imp1/example_main.txt "
2. interval in which variables locate in response to following:"Please enter interval of variables and separate them by comma like -1000, 1000: "
3. step through which algorithms can precede in response to following: "Please enter the step through which variables can change like 0.1: "

OUTPUTS:
1. Hill climbing time
2. Hill climbing cost
3. Hill climbing final state
4. Simulated annealing time
5. Simulated annealing cost
6. Simulated annealing final state
7. plot of steps
8. plot of comparision of algorithms

CONFIGURATIONS
COST_BOUND = 0.01; an error which we can tolerate for final answer
####################################################################
DIFFERENCE_BOUND = 0.0000000000000001; difference which we can ignore for two successive states for hill climbing algorithm
####################################################################
NUM_OF_ITERATIONS = 1000000; max num of iteration for both algorithms
####################################################################
INITIAL_STEP = 1; a step though which we can pre process the first random state
####################################################################
