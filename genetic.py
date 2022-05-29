# This is to prove it is part of a GIT repository now
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

# Initialising the variables that are used in developing the genetic algorithm
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

# This generates the size of the genome that is used in the algorithm
# Returns choices
# Parameters:
# @length = Represents the length of the genome as an int
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

# This generates population that fills the genome that is used in the algorithm
# Returns the filled genome list
# Parameters:
# @size = Represents the size of the population as an int
# @genome_length = Represents the length of the genome that is being filled as an int
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

# This causes the crossover event that leads into the mutation
# Returns Genomes a and b after the crossover has occurred
# Parameters:
# @a = Genome a
# @b = Genome b
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

# This causes the mutation that allows for the evolution of the genome population
# Returns the mutated genome
# @genome = Represents the genome
# @num = Represents the number of mutations being done as int = 1
# @probability = Represents the probability of the genome being mutated as a float
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

# This sets the fitness of the population of genomes that were generated
# Returns the fitness function of the genomes
# Parameters:
# @population = Represents the population of genomes that are being rated
# @fitness_func = Represents the fitness function that is being used to rate the genome
def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])

# This chooses two genomes from the population that will be used in the crossover and mutation
# Returns the two that were chosen based on their respective ratings
# Parameters:
# @population = Represents the population of genomes that the two are being chosen from
# @fitness_func = Represents the fitness function that has been used to rate the genome
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )

# This sorts through the population of genomes based on the rating given to them from the fitness function from highest rated to lowest (could be wrong)
# Returns the sorted population object
# Parameters;
# @population = Represents the population of genomes that are being sorted
# @fitness_func = Represents the fitness function that helps with the sorting
def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

# Converts the genome object into a list of strings
# Returns the converted genome list
# Parameters:
# @genome: Represents the genome that is being converted
def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

# Prints the information of the population that had been generated
# Returns the sorted population from worst to best
# Parameters:
# @population = Represents the population of genomes
# @generation_id = Represents the generation number as an int
# @fitness_func = Represents the fitness function of said population
def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]

# This function evolves the genome
# Returns the genome population and i (Where i is the genome in the population that is being currently evolved)
# Parameters:
# @populate_func = Represents the callable population of genomes (Callable[[], Population])
# @fitness_func = Represents the callable fitness function (Callable[[Genome], int])
# @fitness_limit = Represents the limit of the fitness function as an int
# @selection_func = Represents the callable selection function that selects from the population, which genomes to crossover and mutate (Callable[[Population, FitnessFunc], Tuple[Genome, Genome]])
# @crossover_func = Represents the callable crossover function (Callable[[Genome, Genome], Tuple[Genome, Genome]])
# @mutation_func = Represents the callable mutation function (Callable[[Genome], Genome])
# @generation_limit = Represents the maximum number of generations that the algorithm can go through before it stops evolving as an int
# @printer = Represents the callable printer function that is used to output the results (Callable[[Population, int, FitnessFunc], None])
def run_evolution(populate_func: PopulateFunc, fitness_func: FitnessFunc, fitness_limit: int, selection_func: SelectionFunc = selection_pair, crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation, generation_limit: int = 100, printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i