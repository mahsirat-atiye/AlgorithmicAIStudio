import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import time

COST_BOUND = 0.01
DIFFERENCE_BOUND = 0.0000000000000001
NUM_OF_ITERATIONS = 1000000
INITIAL_STEP = 1
hill_climbing_costs = []
annealing_costs = []


def read_data(address):
    f = open(address, "r")
    if f.mode == 'r':
        #  to check that the file is in open mode
        fl = f.readlines()
        num_of_equations = len(fl)
        num_of_variable = len(fl[0].split(",")) - 1
        A = [[0.0] * num_of_variable for i in range(num_of_equations)]
        b = [0.0] * num_of_equations
        for i, equation in enumerate(fl):
            equation_ = equation.split(",")
            for j in range(num_of_variable):
                A[i][j] = float(equation_[j])
            b[i] = float(equation_[num_of_variable].rstrip())
        return A, b, num_of_variable


def compute_cost(A, b, X):
    err = 0
    for j, equation in enumerate(A):
        s = 0
        for i in range(len(X)):
            s += (equation[i] * X[i])
        err += (s - b[j]) ** 2
    return err


def generate_random_state(t1, t2, num_of_variables):
    state = [0.0] * num_of_variables
    for i in range(num_of_variables):
        state[i] = random.uniform(t1, t2)

    return state


def find_best_neighbor(A, b, current_state, step, t1, t2):
    current_state_cost = compute_cost(A, b, current_state)
    best_neighbor_state = current_state
    best_neighbor_state_cost = current_state_cost
    for i in range(len(current_state)):
        temp = copy.copy(current_state)
        temp[i] += step
        if t2 >= temp[i] >= t1:
            new_cost = compute_cost(A, b, temp)
            if new_cost < best_neighbor_state_cost:
                best_neighbor_state = copy.copy(temp)
                best_neighbor_state_cost = new_cost
        temp[i] -= 2 * step
        if t2 >= temp[i] >= t1:
            new_cost = compute_cost(A, b, temp)
            if new_cost < best_neighbor_state_cost:
                best_neighbor_state = copy.copy(temp)
                best_neighbor_state_cost = new_cost

    return best_neighbor_state, best_neighbor_state_cost, current_state_cost - best_neighbor_state_cost


def hill_climbing(A, b, step, current_state, num_of_iterations):
    current_state_cost = compute_cost(A, b, current_state)
    if current_state_cost < COST_BOUND:
        return current_state

    for i in range(num_of_iterations):

        next_sate, next_cost, diff = find_best_neighbor(A, b, current_state, step, t1, t2)
        if diff < DIFFERENCE_BOUND:
            # print("CASE1; No huge difference detected by this step: ", step)
            return next_sate, next_cost
        if next_cost < COST_BOUND:
            # print("CASE2; Almost reached the optimal solution")
            return next_sate, next_cost
        current_state = next_sate
        current_state_cost = next_cost
        hill_climbing_costs.append(current_state_cost)

    # print("CASE3; Iteration limit exceeded")
    return current_state, current_state_cost


def solution_hill_climbing(t1, t2, num_of_variables, A, b, step):
    current_state = generate_random_state(t1, t2, num_of_variables)
    current_state_cost = compute_cost(A, b, current_state)
    hill_climbing_costs.append(current_state_cost)
    # print("random state & cost: ")
    # print(current_state, current_state_cost)

    current_state, current_state_cost = hill_climbing(A, b, INITIAL_STEP, current_state, num_of_variables * (int(t2) - int(t1)))
    # current_state, current_state_cost = hill_climbing(A, b, 1, current_state, 2 * (int(t2) - int(t1)))
    # print("preprocessed state & cost: ")
    # print(current_state, current_state_cost)

    return hill_climbing(A, b, step, current_state, NUM_OF_ITERATIONS)


def up_down():
    if random.random() < 0.5:
        return True
    return False


def temperature(fraction):
    return max(0.01, min(1, 1 - fraction))


def find_next_neighbor(A, b, current_state, step, t1, t2, temperature):
    current_state_cost = compute_cost(A, b, current_state)
    i = random.randrange(0, num_of_variables, 1)
    temp = copy.copy(current_state)
    if up_down():
        temp[i] += step
    else:
        temp[i] -= step

    if t2 >= temp[i] >= t1:
        new_cost = compute_cost(A, b, temp)
        if new_cost < current_state_cost:
            return temp, new_cost, current_state_cost - new_cost
        if random.random() < np.exp((current_state_cost - new_cost) / temperature):
            return temp, new_cost, current_state_cost - new_cost
    return current_state, current_state_cost, 0


def annealing(A, b, step, current_state, num_of_iterations):
    current_state_cost = compute_cost(A, b, current_state)
    if current_state_cost < COST_BOUND:
        return current_state


    for i in range(num_of_iterations):
        fraction = i / float(num_of_iterations)
        T = temperature(fraction)
        next_sate, next_cost, diff = find_next_neighbor(A, b, current_state, step, t1, t2, T)
        if next_cost < COST_BOUND:
            # print("CASE1; Almost reached the optimal solution")
            return next_sate, next_cost
        current_state = next_sate
        current_state_cost = next_cost
        annealing_costs.append(current_state_cost)

    # print("CASE3; Iteration limit exceeded")
    return current_state, current_state_cost


def solution_annealing(t1, t2, num_of_variables, A, b, step):
    current_state = generate_random_state(t1, t2, num_of_variables)
    current_state_cost = compute_cost(A, b, current_state)
    annealing_costs.append(current_state_cost)
    # print("random state & cost: ")
    # print(current_state, current_state_cost)

    current_state, current_state_cost = annealing(A, b, INITIAL_STEP, current_state, num_of_variables * (int(t2) - int(t1)))
    # current_state, current_state_cost = hill_climbing(A, b, 1, current_state, 2 * (int(t2) - int(t1)))
    # print("preprocessed state & cost: ")
    # print(current_state, current_state_cost)

    return annealing(A, b, step, current_state, NUM_OF_ITERATIONS)


# main
A, b, num_of_variables = read_data(input("Please enter address of txt file like: imp1/example_main.txt "))
t1, t2 = list(
    map(float, input("Please enter interval of variables and separate them by comma like -1000, 1000: ").split(", ")))
step = float(input("Please enter the step through which variables can change like 0.1: "))
# hill climbing
start_time_hill_climbing = time.clock()
hill_climbing_sol = solution_hill_climbing(t1, t2, num_of_variables, A, b, step)
end_time_hill_climbing = time.clock()
print(end_time_hill_climbing - start_time_hill_climbing, " seconds for hill climbing")
print("hill climbing cost: ", hill_climbing_sol[1])
print("hill climbing final state: ", hill_climbing_sol[0])
# annealing
start_time_annealing = time.clock()
annealing_sol = solution_annealing(t1, t2, num_of_variables, A, b, step)
end_time_annealing = time.clock()
print(end_time_annealing - start_time_annealing, " seconds for annealing")
print("annealing cost: ", annealing_sol[1])
print("annealing final state: ", annealing_sol[0])

# plot the results
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('hill climbing vs annealing')

# hill climbing
x_hill_climbing = np.linspace(start=1, stop=len(hill_climbing_costs), num=len(hill_climbing_costs))
y_hill_climbing = np.array(hill_climbing_costs)
# ax1.set_title('cost - iteration in hill climbing')
ax1.set_xlabel('iteration in hill climbing')
ax1.set_ylabel('cost')
ax1.plot(x_hill_climbing, y_hill_climbing, color='blue', linewidth=1, linestyle='--', marker='*', markersize=1)

# annealing
x_annealing = np.linspace(start=1, stop=len(annealing_costs), num=len(annealing_costs))
y_annealing = np.array(annealing_costs)
# ax2.set_title('cost - iteration in simulated annealing')
ax2.set_xlabel('iteration in annealing')
ax2.set_ylabel('cost')
plt.plot(x_annealing, y_annealing, color='red', linewidth=1, linestyle='--', markersize=1)
ax2.plot(x_annealing, y_annealing)

fig1, (ax3, ax4) = plt.subplots(2)
fig1.suptitle('hill climbing vs annealing')

# time comparison
tm = ['Hill climbing', 'Annealing']
ts = [end_time_hill_climbing - start_time_hill_climbing, end_time_annealing - start_time_annealing]
ax3.bar(tm, ts, color='r')
ax3.legend(labels=['Time'])

# cost comparison
cm = ['Hill climbing', 'Annealing']
cs = [hill_climbing_sol[1], annealing_sol[1]]
ax4.bar(cm, cs, color='g')
ax4.legend(labels=['Cost'])
plt.show()

