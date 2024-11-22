# %%
import random
import numpy as np
import multiprocessing
# from objproxies import CallbackProxy
from deap import base, creator, tools, algorithms  # https://deap.readthedocs.io/en/master/

# uncomment cases to test the algorithm
"""
biggest example | HIGHSCORE: 121
to run this faster, run in your terminal
`python product_scheduling.py`
"""
PROCESSES = ['Assembly', 'Testing', 'Packaging', 'Loading']
PROCESS_TIMES = {
    'Cookie': {
        'Assembly': 2,
        'Testing': 1,
        'Packaging': 1,
        'Loading': 1
    },
    'EV car': {
        'Assembly': 10,
        'Testing': 2,
        'Packaging': 1,
        'Loading': 1
    },
    'Hose': {
        'Assembly': 1,
        'Testing': 2,
        'Packaging': 2,
        'Loading': 2
    },
    'Plumbus': {
        'Assembly': 4,
        'Testing': 5,
        'Packaging': 2,
        'Loading': 2
    },
    'Bomb': {
        'Assembly': 1,
        'Testing': 4,
        'Packaging': 5,
        'Loading': 2
    },
    'Cake': {
        'Assembly': 3,
        'Testing': 1,
        'Packaging': 2,
        'Loading': 1
    },
    'Bolts': {
        'Assembly': 1,
        'Testing': 1,
        'Packaging': 1,
        'Loading': 1
    }
}
DEMAND = {
    'Cookie': 15,
    'EV car': 10,
    'Hose': 14,
    'Plumbus': 7,
    'Bomb': 7,
    'Cake': 7,
    'Bolts': 20
}
MACHINES = {'Assembly': 22, 'Testing': 15, 'Packaging': 13, 'Loading': 13}
WORK_HOURS = 12
TIME_SLOT_DURATION = 5
###
"""
variables
"""
# total time slots available per day
TIME_SLOTS = int(WORK_HOURS * 60 / TIME_SLOT_DURATION)

# genetic algorithmp parameters
ERROR_PENALTY = 10000
# POP_SIZE = 200
# CXPB, MUTPB, NGEN = 0.95, 0.675, 5000  # crossover probability, mutation probability, and number of generations
# TOURNAMENT_SIZE = 5
# N_EVALS = 0
# N_GENS = 0
"""
find the earliest time that a process can be in
example:
    'Product 1': {
        'Assembly': 2,
        'Testing': 1,
        'Packaging': 1,
    },

    Hence, earliest possible time for 'Assembly' is time slot 1
    earliest possible time for 'Testing' is time slot 3, (1 + 2)
    earliest possible time for 'Packaging' is time slot 4, (1 + 2 + 1)
"""
process_lag = {}
for pd in PROCESS_TIMES:
    process_lag[pd] = {}
    for p in PROCESS_TIMES[pd]:
        process_lag[pd][p] = 0
        for i in range(PROCESSES.index(p)):
            process_lag[pd][p] += PROCESS_TIMES[pd][PROCESSES[i]]

total_products = 0
for p in DEMAND:
    total_products += DEMAND[p]

total_machines = 0
for m in MACHINES:
    total_machines += MACHINES[m]

# define fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# individual generator
def create_individual():
    eps = 1.5
    schedule = []
    for product in PROCESS_TIMES:
        for process in PROCESS_TIMES[product]:
            for _ in range(DEMAND[product]):
                machine = random.randint(0, MACHINES[process] - 1)

                lowest_index = process_lag[product][process]
                highest_index = TIME_SLOTS - PROCESS_TIMES[product][process]

                rng = int(highest_index * (PROCESSES.index(process) + eps) /
                          (len(PROCESSES)))

                time_slot = random.randint(
                    lowest_index, min(lowest_index + rng, highest_index))

                schedule.append((product, process, machine, time_slot))
    return schedule


toolbox.register("individual", tools.initIterate, creator.Individual,
                 create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# fitness function to minimize makespan
"""
PSEUDOCODE
1. Initialize PENALTY=0, END_TIMES=[], EMPTY_MACHINES=0, PRODUCTS_WAITING=0, COMPLETED=[], MACHINE_STATE=[]
2. SORT all process in the indivisual according to TIME_SLOT
3. FOR EACH individual:
4.      CALCULATE END_TIME = START_TIME + DURATION
5.      INCREMENT the TIME_SLOT of products that are waiting for next process in COMPLETED
6.          ADD amount of products that are waiting to PRODUCTS_WAITING
7.      IF there is product that has completed previous process:
8.          DECREMENT the product from COMPLETED
9.      ELSE:
10.          ADD 10000 to PENALTY
11.     IF machine in current TIME_SLOT is not occupied (i.e. 0):
12.         UPDATE MACHINE_STATE at TIME_SLOT with 1
13.     ELSE:
14.         ADD 10000 to PENALTY
15.     FROM PRODUCT_LAG to END_TIME:
16.         IF MACHINE_STATE for the process equal 0:
17.             EMPTY_MACHINES += 1
18.     UPDATE END_TIMES list with END_TIME
19. REPEAT FOR all processes in individual
20. MAKESPAN = max(END_TIMES) + PENALTY
21. RETURN MAKESPAN, EMPTY_MACHINE, PRODUCTS_WAITING
"""


def evaluate(individual):
    # global N_EVALS, N_GENS
    # N_EVALS += 1
    # if N_EVALS % POP_SIZE == 0:
    #     N_GENS += 1

    # used to hold penalty when process occur before previous process is done
    penalty = 0

    # hold all end time of processes
    end_times = [0] * (TIME_SLOTS + 1)

    # hold completed processes, indexed by time slot
    comp_process = {
        product: [[0] * len(PROCESS_TIMES[product])
                  for _ in range(TIME_SLOTS + 1)]
        for product in PROCESS_TIMES
    }

    # hold machine state, 0 for empty, 1 for occupied
    machine_state = {
        machine: [[0] * MACHINES[machine] for _ in range(TIME_SLOTS + 1)]
        for machine in MACHINES
    }

    # keep track of time slot increment, bring forward all items that have not proceeded to next process
    prev_time = 0

    # sort the schedule according to time slot
    individual = sorted(individual, key=lambda x: x[3])

    # hold fitness score for empty machine slots before a time slot
    empty_machines = 0

    # hold fitness score for items that are left waiting for next process
    products_waiting = 0

    for item in individual:
        product, process, machine, start_time = item

        duration = PROCESS_TIMES[product][process]
        end_time = start_time + duration  # get end time

        ###
        """
        increment the completed products, waiting for the next step of processing
        all the products that have not been consumed for the next process will be brought forward
        """
        if (prev_time != start_time):
            for itm in PROCESS_TIMES:
                for prc in range(len(PROCESS_TIMES[itm])):
                    prev_items = comp_process[itm][prev_time][prc]
                    cur_items = comp_process[itm][start_time][prc]

                    if (prc != len(PROCESSES) - 1):
                        products_waiting += prev_items

                    comp_process[itm][start_time][prc] = cur_items + prev_items

            prev_time = start_time

        ###
        """
        check if there are available product from the previous steps
        since products need to follow the process sequence, need to check if there are any from previous process
        
        if process index is 0, that means the first process, then add 1 to the index
        else deduct one from previous, and add one to current

        when indexing we index with start time, when adding we add to end time
        """
        for ps_idx, ps in enumerate(PROCESSES):
            if (process == ps and ps_idx == 0):
                comp_process[product][end_time][0] += 1
            elif (process == ps):
                if (comp_process[product][start_time][ps_idx - 1] > 0):
                    comp_process[product][start_time][ps_idx - 1] -= 1
                    comp_process[product][end_time][ps_idx] += 1
                else:
                    penalty += ERROR_PENALTY

        ###
        """
        checks if current machine spot is in use
        if 0 then it is not in use, proceed mark array as 1 for number of time slots to process
        exp. 
            process start = 2 
            process time = 3
            machine = 1

            using 0 indexing
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0]
            ]

            since process takes 3 slots to complete, starting from slot 2

        if the machine is already occupied, give the individual (genome/sequence) a heavy penalty,
        since this means it is trying to occupy a machine in use
        """
        if (machine_state[process][start_time][machine] == 0):
            lim = PROCESS_TIMES[product][process]
            for i in range(lim):
                machine_state[process][start_time + i][machine] = 1
        else:
            penalty += ERROR_PENALTY

        ###
        """
        check for empty machine spots from time slot 0 to current start time
        any spots marked with 0 is considered empty
        motivation of this fitness score is to encourage earlier spots to be filled in first
        """
        # for ts in range(max(0, start_time - duration), start_time):
        for ts in range(process_lag[product][process], start_time):
            for i in range(MACHINES[process]):
                if (machine_state[process][ts][i] == 0):
                    empty_machines += 1

        # list of end times, [0, 0, 2, 3, 0]
        end_times[end_time] = max(end_times[end_time], end_time + 1)

    if (products_waiting == 0 and penalty != 0):
        products_waiting = penalty / 10000

    makespan = max(end_times)
    makespan += penalty
    """
    makespan: total time slots taken to complete the order
    empty_machines: aggregate of empty machines from 0 to current time, of each process
    product_waiting: aggregate of products that are waiting for next process to use them

    all score need to be minimized for optimal solution

    'makespan' is the primary score, since if there is overlap the penalty will be big. 
    once there is no overlap, makespan will have small value that can easily be minimized

    'empty_machine' encourages processes to be moved vertically
    (moved to empty machines in the same time slot)
    once 'makespan' stagnates, 'empty_machine' can continue to be optimized
    leading to a more optimal solution and potentially decreasing 'makespan'

    'products_waiting' encourages processes to be moved horizontally
    (moved to earlier time slots, to prevent waiting time)
    once 'makespan' stagnates, 'product_waiting' can continue to be optimized
    leading to a more optimal solution and potentially decreasing 'makespan'
    """
    return (makespan, empty_machines, products_waiting)


"""
PSEUDOCODE
1. INITIALIZE SIZE = length of individual process list
2. SELECT 2 random integers in between 1 and (SIZE - 1) and set to POINT_1 and POINT_2
3. SWAP the intergers IF the POINT_1 is greter than POINT_2
4. FROM POINT_1 to POINT_2:
5.      SWAP only TIME_SLOT values, only MACHINE number or BOTH values based on random probability
6. RETURN both individuals
"""


def cxSelectiveTwoPoint(ind1, ind2):
    # choose crossover points
    size = len(ind1)
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)

    # ensure cxpoint1 is less than cxpoint2
    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # swap the `machine` and `time_slot` between the two individuals from cxpoint1 to cxpoint2
    for i in range(cxpoint1, cxpoint2):
        swappb = random.randint(0, 1)

        # keep `product` and `process` constant
        product1, process1, machine1, time_slot1 = ind1[i]
        product2, process2, machine2, time_slot2 = ind2[i]

        if swappb < 0.6:
            # swap `time_slot` values only
            ind1[i] = (product1, process1, machine1, time_slot2)
            ind2[i] = (product2, process2, machine2, time_slot1)
        elif swappb < 0.95:
            # swap `machine` values only
            ind1[i] = (product1, process1, machine2, time_slot1)
            ind2[i] = (product2, process2, machine1, time_slot2)
        else:
            # swap `machine` and `time_slot` values only
            ind1[i] = (product1, process1, machine2, time_slot2)
            ind2[i] = (product2, process2, machine1, time_slot1)

    return ind1, ind2


def cxSelectiveOnePoint(ind1, ind2):
    # choose crossover points
    size = len(ind1)
    cxpoint = random.randint(0, size - 1)

    # swap the `machine` and `time_slot` between the two individuals from cxpoint1 to cxpoint2
    swappb = random.randint(0, 1)

    # keep `product` and `process` constant
    product1, process1, machine1, time_slot1 = ind1[cxpoint]
    product2, process2, machine2, time_slot2 = ind2[cxpoint]

    if swappb < 0.6:
        # swap `time_slot` values only
        ind1[cxpoint] = (product1, process1, machine1, time_slot2)
        ind2[cxpoint] = (product2, process2, machine2, time_slot1)
    elif swappb < 0.95:
        # swap `machine` values only
        ind1[cxpoint] = (product1, process1, machine2, time_slot1)
        ind2[i] = (product2, process2, machine1, time_slot2)
    else:
        # swap `machine` and `time_slot` values only
        ind1[cxpoint] = (product1, process1, machine2, time_slot2)
        ind2[cxpoint] = (product2, process2, machine1, time_slot1)

    return ind1, ind2


"""
PSEUDOCODE
1. FOR EACH process in the individual:
2.      IF random probability < MUTATION_RATE:
3.          EITHER MUTATE by randomly assigning MACHINE, INCREMENT or DECREMENT MACHINE index
4.      IF random pobability < MUTATION_RATE:
5.          EITHER MUTATE by randomly assigning TIME_SLOT, INCREMENT or DECREMENT TIME_SLOT index
6. RETURN individual
"""


def mutate(individual, indpb):
    for i in range(len(individual)):
        # unpack the current schedule entry
        product, process, machine, time_slot = individual[i]

        # apply mutation based on the probability `indpb`
        if random.random() < indpb:
            """
            20 % for randomly distributing machines, initially to converge at a valid configuration
            40 % for both moving the machine up or down by 1, encourage vertical movement
            """
            # mutate the machine assignment
            pb = random.random()
            if pb < 0.2:
                machine = random.randint(0, MACHINES[process] - 1)
            elif pb < 0.6:
                machine = (machine - 1) % MACHINES[process]
            else:
                machine = (machine + 1) % MACHINES[process]

        if random.random() < indpb:
            """
            10 % for randomly distributing time slots, initially to converge at a valid configuration
            45 % for both moving the time slot up or down by 1, encourage horizontal movement
            """
            # mutate the time slot assignment
            pb = random.random()
            if pb < 0.1:
                time_slot = random.randint(
                    process_lag[product][process],
                    TIME_SLOTS - PROCESS_TIMES[product][process])
            elif pb < 0.55:
                time_slot = max(
                    time_slot - 1,
                    process_lag[product][process],
                )
            else:
                time_slot = min(time_slot + 1,
                                TIME_SLOTS - PROCESS_TIMES[product][process])

        # update the individual's schedule with the mutated values
        individual[i] = (product, process, machine, time_slot)

    return (individual, )


from colorama import Fore  # for color text in terminal/notebook


# print function to display in terminal/notebook and write to log.txt file
def printSchedule(schedule):
    write_str = ''

    makespan, empty_machine, products_waiting = evaluate(schedule)

    schedule = sorted(schedule, key=lambda x: x[3])

    process_count = {
        product: [[0] * len(PROCESS_TIMES[product])
                  for _ in range(TIME_SLOTS + 1)]
        for product in PROCESS_TIMES
    }

    machine_state = {
        machine: [['e'] * MACHINES[machine] for _ in range(TIME_SLOTS + 1)]
        for machine in MACHINES
    }

    prev_time = 0

    for item in schedule:
        product, process, machine, start_time = item

        # increment the completed products, waiting for the next step of processing
        if (prev_time != start_time):
            for time in range(1, start_time - prev_time + 1):
                for itm in PROCESS_TIMES:
                    for prc in range(len(PROCESS_TIMES[itm])):
                        process_count[itm][
                            prev_time + time][prc] = process_count[itm][
                                prev_time +
                                time][prc] + process_count[itm][prev_time +
                                                                time - 1][prc]

            prev_time = start_time

        duration = PROCESS_TIMES[product][process]
        end_time = start_time + duration

        # check if there are available product from the previous steps
        for ps_idx, ps in enumerate(PROCESSES):
            if (process == ps and ps_idx == 0):
                process_count[product][end_time][0] += 1
            elif (process == ps):
                if (process_count[product][start_time][ps_idx - 1] > 0):
                    process_count[product][end_time][ps_idx] += 1

        # check for machines that are currently in use
        if (machine_state[process][start_time][machine] == 'e'):
            lim = PROCESS_TIMES[product][process]
            for i, p in enumerate(PROCESS_TIMES):
                if (product == p):
                    for j in range(lim):
                        machine_state[process][start_time +
                                               j][machine] = f"P{i+1}"

    # increment the completed products, waiting for the next step of processing
    start_time = prev_time + 1
    if (prev_time != start_time):
        for time in range(1, start_time - prev_time + 1):
            for itm in PROCESS_TIMES:
                for prc in range(len(PROCESS_TIMES[itm])):
                    process_count[itm][prev_time + time][prc] = process_count[
                        itm][prev_time +
                             time][prc] + process_count[itm][prev_time + time -
                                                             1][prc]

        prev_time = start_time

    schedule = np.asarray(schedule)
    for i in range(schedule.shape[0]):
        schedule[i][2] = str(int(schedule[i][2]) + 1)
        schedule[i][3] = str(int(schedule[i][3]) + 1)

    # print
    print(Fore.WHITE + '')

    print(Fore.WHITE + "-- PRODUCT DETAILS --")
    write_str += "-- PRODUCT DETAILS --\n"
    for i, product in enumerate(PROCESS_TIMES):
        print(Fore.GREEN + f"{i+1}. {product}")
        write_str += f"{i+1}. {product}\n"
        for process in PROCESS_TIMES[product]:
            print(Fore.LIGHTYELLOW_EX +
                  f"\t{process}: {PROCESS_TIMES[product][process]}")
            write_str += f"\t{process}: {PROCESS_TIMES[product][process]}\n"

    print(Fore.WHITE + "\n-- MACHINE TYPES --")
    write_str += "\n-- MACHINE TYPES --\n"
    for i, process_type in enumerate(MACHINES):
        print(Fore.LIGHTBLUE_EX +
              f"{i+1}. {process_type}: {MACHINES[process_type]}")
        write_str += f"{i+1}. {process_type}: {MACHINES[process_type]}\n"

    print(Fore.WHITE + "\n-- TIME SLOTS --")
    write_str += "\n-- TIME SLOTS --\n"
    print(Fore.CYAN +
          f"{TIME_SLOTS} time slots. {TIME_SLOT_DURATION} mins each")
    write_str += f"{TIME_SLOTS} time slots. {TIME_SLOT_DURATION} mins each\n"

    print(Fore.WHITE + "\n-- SCHEDULE FORMAT --")
    write_str += "\n-- SCHEDULE FORMAT --\n"
    print(Fore.WHITE + "[ " + Fore.GREEN + "product" + Fore.WHITE + ", " +
          Fore.LIGHTYELLOW_EX + "process" + Fore.WHITE + ", " +
          Fore.LIGHTBLUE_EX + "machine_num" + Fore.WHITE + ", " + Fore.CYAN +
          "time_slot" + Fore.WHITE + " ]")
    write_str += "[ product, process, machine_num, time_slot ]\n"

    print(Fore.WHITE + "\n-- SCHEDULE --")
    print(schedule)
    write_str += "\n-- SCHEDULE --\n"
    for sch in schedule:
        write_str += str(sch) + '\n'
    write_str += '\n'
    print()

    print(f"-- NUMBER OF PRODUCT COMPLETED AT TIME --")
    write_str += f"-- NUMBER OF PRODUCT COMPLETED AT TIME --\n"
    time_slots_header = 'TIME SLOT\t|'
    for i in range(TIME_SLOTS):
        time_slots_header += f"{i+1}\t|"
    print(Fore.CYAN + time_slots_header)
    write_str += time_slots_header + '\n'

    for product in PROCESS_TIMES:
        print(Fore.GREEN + f"\n{product}\t|")
        write_str += f"\n{product}\t|\n"
        for i, process in enumerate(PROCESS_TIMES[product]):
            process_row_str = f"  {process}\t|"
            for ts in range(TIME_SLOTS):
                process_row_str += f"{process_count[product][ts][i]}\t|"
            print(Fore.LIGHTYELLOW_EX + process_row_str)
            write_str += process_row_str + '\n'

    for machine in machine_state:
        print(Fore.LIGHTYELLOW_EX + f"\n{machine}  \t|")
        write_str += f"\n{machine}  \t|\n"
        for i in range(len(machine_state[machine][0])):
            machine_row_str = f"  Machine {i+1}\t|"
            for ts in range(TIME_SLOTS):
                machine_row_str += f"{machine_state[machine][ts][i]}\t|"
            print(Fore.LIGHTBLUE_EX + machine_row_str)
            write_str += machine_row_str + '\n'

    print(Fore.WHITE + f"\n-- Makespan --" + Fore.RED + f"\n   {makespan}")
    write_str += f"\n-- Makespan --" + f"\n   {makespan}\n"

    print(Fore.WHITE + '')

    with open('result.txt', 'w+') as f:
        f.write(write_str)
        f.close()

    if (MODE == 'cx'):
        with open('crossover_biggest_result.txt', 'a+') as f:
            f.write('\n' + write_str + f'\n;\n')
            f.close()

        with open('crossover_biggest.txt', 'a+') as f:
            crs_data = f"{POP_SIZE},{CXPB},{MU_INDPB},{NGEN},{TOURNAMENT_SIZE},{makespan},{empty_machine},{products_waiting}\n"
            f.write(crs_data)
            f.close()

    elif (MODE == 'mu'):
        with open('mutation_biggest_result.txt', 'a+') as f:
            f.write('\n' + write_str + f'\n;\n')
            f.close()

        with open('mutation_biggest.txt', 'a+') as f:
            crs_data = f"{POP_SIZE},{CXPB},{MU_INDPB},{NGEN},{TOURNAMENT_SIZE},{makespan},{empty_machine},{products_waiting}\n"
            f.write(crs_data)
            f.close()

    elif (MODE == 'ngen'):
        with open('ngen_biggest_result.txt', 'a+') as f:
            f.write('\n' + write_str + f'\n;\n')
            f.close()

        with open('ngen_biggest.txt', 'a+') as f:
            crs_data = f"{POP_SIZE},{CXPB},{MU_INDPB},{NGEN},{TOURNAMENT_SIZE},{makespan},{empty_machine},{products_waiting}\n"
            f.write(crs_data)
            f.close()

    elif (MODE == 'pop'):
        with open('pop_biggest_result.txt', 'a+') as f:
            f.write('\n' + write_str + f'\n;\n')
            f.close()

        with open('pop_biggest.txt', 'a+') as f:
            crs_data = f"{POP_SIZE},{CXPB},{MU_INDPB},{NGEN},{TOURNAMENT_SIZE},{makespan},{empty_machine},{products_waiting}\n"
            f.write(crs_data)
            f.close()

    elif (MODE == 'tourn'):
        with open('tourn_biggest_result.txt', 'a+') as f:
            f.write('\n' + write_str + f'\n;\n')
            f.close()

        with open('tourn_biggest.txt', 'a+') as f:
            crs_data = f"{POP_SIZE},{CXPB},{MU_INDPB},{NGEN},{TOURNAMENT_SIZE},{makespan},{empty_machine},{products_waiting}\n"
            f.write(crs_data)
            f.close()


def main():
    # random.seed(42)
    random.seed(10)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

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
    # process pool for parallel processing & multi threading
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    POP_SIZE = 100
    # crossover probability, mutation probability (population percentage), and number of generations
    CXPB, MUTPB, NGEN = 0.7, 0.5, 4000  # MUTPB is kept constant
    MU_INDPB = 0.03  # individual mutation probability
    TOURNAMENT_SIZE = 5

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxSelectiveTwoPoint)
    # toolbox.register("mate", cxSelectiveOnePoint)
    toolbox.register("mutate", mutate, indpb=MU_INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


    MODE = 'ngen'
    MU_INDPB = 0.03
    NGEN = 250
    for i in range(39):
        # main driver
        pop, log, hof = main()
        best_ind = hof.items[0]

        # output results
        printSchedule(best_ind)

        NGEN += 250

    pool.close()

# %%
