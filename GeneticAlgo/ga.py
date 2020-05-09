import random
from math import sin, cos, pi, log
import time
# Configs
POPULATION_SIZE = 100
OPERATIONAL_GENES = '''-+*/^sc'''
RANGE_OF_CONSTANTS = 8
ZERO = 0.0001
PROB_OF_CHOOSING_VARIABLE = 0.3
PROB_OF_MUTATION = 0.1
ELITISM_PARTITION = 0.1
TOP_GENES_PARTITION = 0.5
MAX_NUM_OF_GENERATIONS = 10000
NUM_OF_SAMPLES = 6
# inputs & rest of configurations
target = input("Please enter target formula: like: sin(pi) + 6*x\t")
t1, t2 = list(
    map(float, input("Please enter domain of x and separate them by comma like -1, 5.6: ").split(", ")))
x = []
for i in range(NUM_OF_SAMPLES):
    x.append(random.uniform(t1, t2))
y = list(map(lambda x: eval(target), x))
len_of_tree = input("Please enter length of estimating formula tree, as you choose longer one, it is more probable to wait longer, it has to be 2^n-1: like: 7\t")
N = 2*2 ** int((log(int(len_of_tree), 2))) - 1

error_limit = input("Please enter max error limit, which you can tolerate! like: 0.1\t")
ERROR_LIMIT = float(error_limit)

start_time  = time.clock()


# representing binary tree of formula as an array

def left_child(i):
    return 2 * i + 1


def right_child(i):
    return (2 * i) + 2


def is_parent(i, N):
    if left_child(i) < N or right_child(i) < N:
        return True
    return False


def get_formula_util_(a, root, s):
    if root < len(a):
        # First recur on left child

        get_formula_util_(a, left_child(root), s)

        # then print the data of node
        s.append(a[root])
        # print(a[root], end=''),

        # now recur on right child
        get_formula_util_(a, right_child(root), s)
        # s += ")"
        # print(")", end='')
    s_size = len(s)
    # print(s)
    return s


class Individual(object):

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.fitness()

    @classmethod
    def mutated_genes(self, i):
        global N

        if is_parent(i, N):
            #     if it is parent
            global OPERATIONAL_GENES
            gene = random.choice(OPERATIONAL_GENES)
            return gene
        #     it is child
        global PROB_OF_CHOOSING_VARIABLE
        global RANGE_OF_CONSTANTS
        prob = random.random()
        if prob < PROB_OF_CHOOSING_VARIABLE:
            return 'x'
        return str(random.uniform(-RANGE_OF_CONSTANTS, RANGE_OF_CONSTANTS))

    @classmethod
    def create_gnome(self):

        global N
        gnome_len = N
        return [self.mutated_genes(_) for _ in range(gnome_len)]

    def mate(self, partner):

        child_chromosome = []
        index = 0
        for genes_in_partner1, genes_in_partner2 in zip(self.chromosome, partner.chromosome):

            prob = random.random()

            global PROB_OF_MUTATION
            if prob < (1 - PROB_OF_MUTATION) / 2:
                child_chromosome.append(genes_in_partner1)

            elif prob < (1 - PROB_OF_MUTATION):
                child_chromosome.append(genes_in_partner2)

            # otherwise mutation occurs
            else:
                child_chromosome.append(self.mutated_genes(index))
            index += 1

        return Individual(child_chromosome)

    def get_formula_util_as_string(self, root, s):
        if root < len(self.chromosome):
            s.append("(")
            self.get_formula_util_as_function(left_child(root), s)

            if self.chromosome[root] == 's':
                s.append('* sin')
            elif self.chromosome[root] == 'c':
                s.append('* cos')
            elif self.chromosome[root] == '^':
                s.append('**')
            elif self.chromosome[root] == '0':
                global ZERO
                # as there might be problems of raising zero by negative power or division by zero
                s.append(str(ZERO))
            else:
                s.append(self.chromosome[root])

            self.get_formula_util_as_function(right_child(root), s)
            s.append(")")
        base = ""
        formula_string = base.join(s)
        return formula_string

    def get_formula_util_as_function(self, root, s):
        if root < len(self.chromosome):
            s.append("(")
            self.get_formula_util_as_function(left_child(root), s)

            if self.chromosome[root] == 's':
                s.append('* sin')
            elif self.chromosome[root] == 'c':
                s.append('* cos')
            elif self.chromosome[root] == '^':
                s.append('**')
            elif self.chromosome[root] == '0':
                global ZERO
                # as there might be problems of raising zero by negative power or division by zero
                s.append(str(ZERO))
            else:
                s.append(self.chromosome[root])

            self.get_formula_util_as_function(right_child(root), s)
            s.append(")")
        base = ""
        formula_string = base.join(s)
        f = lambda x: eval(formula_string)
        return f

    def traverse_as_tree(self, i=0, d=0):
        if i >= len(self.chromosome):
            return
        l = left_child(i)
        r = right_child(i)
        self.traverse_as_tree(r, d=d + 1)
        print("   " * d + str(self.chromosome[i]))
        self.traverse_as_tree(l, d=d + 1)
    def inorder_traverse(self, root, s):
        # in order traversing of binary tree
        if root < len(self.chromosome):
            self.inorder_traverse(left_child(root), s)

            s.append(self.chromosome[root])

            self.inorder_traverse(right_child(root), s)

        return s

    def fitness(self):
        global x
        global y
        f = self.get_formula_util_as_function(0, list([]))
        # print(self.get_formula_util_as_list(0, list([])))
        try:
            # as there might be unpredictable values by inserting x in formula
            generated_y = list(map(f, x))
        except:
            return float('inf')

        fitness = 0

        for real_genes, generated_genes in zip(y, generated_y):
            try:
                fitness += abs(real_genes - generated_genes)
            except:
                return float('inf')
        return fitness


def main():
    global POPULATION_SIZE
    generation_index = 1
    found = False
    population = []
    # create initial population
    for i in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:

        # sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)

        # if the individual having proper fitness, the solution is already found!
        global ERROR_LIMIT
        global MAX_NUM_OF_GENERATIONS
        if population[0].fitness <= ERROR_LIMIT or generation_index > MAX_NUM_OF_GENERATIONS:
            found = True
            print("FINAL SOLUTION OF GA AS ‌TREE\n\n")
            population[0].traverse_as_tree(0, 0)
            print("\nGeneration: {}\tformula: {}\tFitness: {}". \
                  format(generation_index, population[0].get_formula_util_as_string(0, list([]))
                         ,
                         population[0].fitness))
            break

        # Otherwise we need new generation to improve fitness
        new_generation = []

        # Elitism
        global ELITISM_PARTITION
        s = int(ELITISM_PARTITION * POPULATION_SIZE)
        # !
        new_generation.extend(population[:s])
        s = int((1 - ELITISM_PARTITION) * POPULATION_SIZE)
        # elites of the society should remain!
        # parentS will be chosen from TOP genes of current population
        global TOP_GENES_PARTITION
        m = int(POPULATION_SIZE * TOP_GENES_PARTITION)
        for i in range(s):
            parent1 = random.choice(population[:m])
            parent2 = random.choice(population[:m])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tformula: {}\tFitness: {}". \
              format(generation_index, population[0].get_formula_util_as_string(0, list([]))
                     ,
                     population[0].fitness))

        generation_index += 1


if __name__ == '__main__':
    main()
    print("TOTAL TIME OF GA: ")
    print(time.clock()-start_time)

