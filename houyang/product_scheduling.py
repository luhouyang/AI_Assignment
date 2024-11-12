# %%
import random
import numpy as np
from deap import base, creator, tools, algorithms

# constants for the problem
PROCESS_TIMES = {
    'Product 1': {
        'Assembly': 2,
        'Testing': 1,
        'Packaging': 1
    },  # time slots required
    'Product 2': {
        'Assembly': 3,
        'Testing': 2,
        'Packaging': 1
    }
}
DEMAND = {'Product 1': 3, 'Product 2': 2}
MACHINES = {'Assembly': 2, 'Testing': 2, 'Packaging': 2}
TIME_SLOTS = 32  # total time slots available per day
ERROR_PENALTY = 1000

# genetic algorithmp parameters
POP_SIZE = 1000
CXPB, MUTPB, NGEN = 0.7, 0.2, 1000  # crossover probability, mutation probability, and number of generations


def biased_randint(low, high, bias_factor):
    # Generate a random number from an exponential distribution
    skewed_num = random.betavariate(bias_factor, 1)

    # Scale the skewed number to the desired range
    return low + int(skewed_num * (high - low))


# define fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# individual generator
def create_individual():
    schedule = []
    for product in PROCESS_TIMES:
        for process in PROCESS_TIMES[product]:
            for _ in range(DEMAND[product]):
                machine = random.randint(0, MACHINES[process] - 1)
                # time_slot = random.randint(0, TIME_SLOTS - PROCESS_TIMES[product][process])
                time_slot = biased_randint(
                    0, TIME_SLOTS - PROCESS_TIMES[product][process], 0.5)
                schedule.append((product, process, machine, time_slot))
    return schedule


toolbox.register("individual", tools.initIterate, creator.Individual,
                 create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# fitness function to minimize makespan
def evaluate(individual):
    penalty = 0
    end_times = [0] * (TIME_SLOTS + 1)

    process_count = {
        product: [[0] * len(PROCESS_TIMES[product])
                  for _ in range(TIME_SLOTS + 1)]
        for product in PROCESS_TIMES
    }

    machine_state = {
        machine: [[0] * MACHINES[machine] for _ in range(TIME_SLOTS + 1)]
        for machine in MACHINES
    }

    prev_time = 0
    individual = sorted(individual, key=lambda x: x[3])

    for item in individual:
        product, process, machine, start_time = item

        # increment the completed products, waiting for the next step of processing
        if (prev_time != start_time):
            for itm in PROCESS_TIMES:
                for prc in range(len(PROCESS_TIMES[itm])):
                    process_count[itm][start_time][prc] = process_count[itm][
                        start_time][prc] + process_count[itm][prev_time][prc]

            prev_time = start_time

        duration = PROCESS_TIMES[product][process]
        end_time = start_time + duration

        # check if there are available product from the previous steps
        if (process == 'Assembly'):
            process_count[product][end_time][0] += 1
        elif (process == 'Testing'):
            if (process_count[product][start_time][0] > 0):
                process_count[product][start_time][0] -= 1
                process_count[product][end_time][1] += 1
            else:
                penalty += ERROR_PENALTY
        elif (process == 'Packaging'):
            if (process_count[product][start_time][1] > 0):
                process_count[product][start_time][1] -= 1
                process_count[product][end_time][2] += 1
            else:
                penalty += ERROR_PENALTY

        # check for machines that are currently in use
        if (machine_state[process][start_time][machine] == 0
                or machine_state[process][start_time][machine] == 2):
            lim = PROCESS_TIMES[product][process]
            for i in range(lim):
                machine_state[process][start_time + i][machine] = 1
            if (start_time + lim + 1 < TIME_SLOTS):
                machine_state[process][start_time + lim + 1][machine] = 2
        else:
            penalty += ERROR_PENALTY

        # list of end times, [0, 0, 2, 3, 0]
        end_times[end_time] = max(end_times[end_time], end_time)

    makespan = max(end_times)
    makespan += penalty
    return (makespan, )


def cxSelectiveTwoPoint(ind1, ind2):
    # Choose crossover points
    size = len(ind1)
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)

    # Ensure cxpoint1 is less than cxpoint2
    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Swap the `machine` and `time_slot` between the two individuals from cxpoint1 to cxpoint2
    for i in range(cxpoint1, cxpoint2):
        # Keep `product` and `process` constant
        product1, process1, machine1, time_slot1 = ind1[i]
        product2, process2, machine2, time_slot2 = ind2[i]

        # Swap `machine` and `time_slot` values only
        ind1[i] = (product1, process1, machine2, time_slot2)
        ind2[i] = (product2, process2, machine1, time_slot1)

    return ind1, ind2


def mutate(individual, indpb=0.05):
    for i in range(len(individual)):
        # Unpack the current schedule entry
        product, process, machine, time_slot = individual[i]

        # Apply mutation based on the probability `indpb`
        if random.random() < indpb:
            # Mutate the machine assignment
            machine = random.randint(0, MACHINES[process] - 1)

        if random.random() < indpb:
            # Mutate the time slot
            time_slot = random.randint(
                0, TIME_SLOTS - PROCESS_TIMES[product][process])

        # Update the individual's schedule with the mutated values
        individual[i] = (product, process, machine, time_slot)

    return (individual, )


toolbox.register("evaluate", evaluate)
toolbox.register("mate", cxSelectiveTwoPoint)
toolbox.register("mutate", mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(42)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=NGEN,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    return pop, log, hof


if __name__ == '__main__':
    pop, log, hof = main()
    best_ind = hof.items[0]
    print("Best schedule:")
    print(np.asarray(sorted(best_ind, key=lambda x: x[3])))
    # print(np.asarray(best_ind))
    print(f"With makespan: {evaluate(best_ind)[0]}")

# %%
