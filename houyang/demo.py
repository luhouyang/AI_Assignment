# %%
"""
string matching Genetic Algorithm
"""

import random


# class that performs initialization, mutation, crossover, fitness calculation
class Chromosome(object):

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.fitness_score()

    # mutation mechanism
    @classmethod
    def genes_mutated(self):
        global GENES
        random_gene = random.choice(GENES)
        return random_gene

    # create a chromosome
    @classmethod
    def create_chromosome(self):
        global SOLUTION
        genome_len = len(SOLUTION)
        return [self.genes_mutated() for _ in range(genome_len)]

    # crossover mechanism
    def crossover(self, second_parent):
        child_chromosome = []
        for genome_first, genome_second in zip(self.chromosome,
                                               second_parent.chromosome):
            prob = random.random()
            if prob < 0.25:
                child_chromosome.append(genome_first)
            elif prob < 0.95:
                child_chromosome.append(genome_second)
            else:
                child_chromosome.append(self.genes_mutated())
        return Chromosome(child_chromosome)

    # fitness function
    # each character in string that doesn't match +1
    # lower value is better
    def fitness_score(self):
        global SOLUTION
        fitness = 0
        for xx, yy in zip(self.chromosome, SOLUTION):
            # if xx != yy: fitness += 1
            if xx != yy: fitness += abs(GENES.find(yy) - GENES.find(xx))
        return fitness


POPULATION_SIZE = 300
GENES = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789, !-[]{}().;:_=&'#%/\\?@~$<>"""
SOLUTION = "hei hei De Hei Ren {=.=} [>.<] \\('_')/ wo men shI Z0NG GU0 ren ([{!-.-!}])"


def main():
    population = []
    solution_found = False
    generation = 1

    # create initial population
    for i in range(POPULATION_SIZE):
        genome = Chromosome.create_chromosome()
        population.append(Chromosome(genome))

    while not solution_found:
        # sort the choromosomes in increasing order
        population = sorted(population, key=lambda x: x.fitness)

        # print top 5 of current generation
        print(f"Generation {generation}:")
        for g in population[:3]:
            print(f"Chromosome: {"".join(g.chromosome)} | Fitness: {g.fitness}")
        print()

        # termination condition
        if population[0].fitness <= 0:
            solution_found = True
            break

        # if not terminated, create new generation
        new_generation = []

        # selection, top 5%
        x = int((50 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:x])

        # crossover and mutation
        x = int((70 * POPULATION_SIZE) / 100)
        for _ in range(x):
            first_parent = random.choice(new_generation)
            second_parent = random.choice(population[:50])

            child = first_parent.crossover(second_parent)
            new_generation.append(child)

        population = new_generation
        generation += 1


if __name__ == '__main__':
    main()

# %%
