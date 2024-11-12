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
DEMAND = {'Product 1': 10, 'Product 2': 8}
MACHINES = {'Assembly': 2, 'Testing': 2, 'Packaging': 2}
TIME_SLOTS = 32  # total time slots available per day

# genetic algorithmp parameters
POP_SIZE = 100
CXPB, MUTPB, NGEN = 0.7, 0.2, 50  # crossover probability, mutation probability, and number of generations

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
                machine = random.randint(1, MACHINES[process])
                time_slot = random.randint(
                    0, TIME_SLOTS - PROCESS_TIMES[product][process])
                schedule.append((product, process, machine, time_slot))
    return schedule


toolbox.register("individual", tools.initIterate, creator.Individual,
                 create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# fitness function to minimize makespan
def evaluate(individual):
    end_times = [0] * (TIME_SLOTS + 1)
    process_count = {
        product: [[0] * len(PROCESS_TIMES[product])
                  for _ in range(TIME_SLOTS + 1)]
        for product in PROCESS_TIMES
    }

    prev_time = 0
    individual = sorted(individual, key=lambda x: x[3])
    
    for item in individual:
        product, process, machine, start_time = item

        # increment the completed products, waiting for the next step of processing
        if (prev_time != start_time):
            for p in PROCESS_TIMES:
                process_count[p][start_time] = process_count[p][
                    start_time] + process_count[p][prev_time]

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
                t = PROCESS_TIMES[product]['Assembly']

                if (end_time + t < TIME_SLOTS):
                    end_time += t
                    process_count[product][end_time][1] += 1
                else:
                    end_time = TIME_SLOTS
        elif (process == 'Packaging'):
            if (process_count[product][start_time][1] > 0):
                process_count[product][start_time][1] -= 1
                process_count[product][end_time][2] += 1
            else:
                t = PROCESS_TIMES[product]['Testing']

                if (end_time + t < TIME_SLOTS):
                    end_time += t
                    process_count[product][end_time][2] += 1
                else:
                    end_time = TIME_SLOTS

        # TODO add a check for machines that are currently in use

        # list of end times, [0, 0, 2, 3, 0]
        end_times[end_time] = max(end_times[end_time], end_time)
    makespan = max(end_times)
    return (makespan, )


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
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
