#   ACTUAL MUSIC GENERATOR. USE ME PLEASE.


from time import sleep
from turtle import color
import click
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from midiutil import MIDIFile
from pyo import *

from genetic import generate_genome, Genome, population_fitness, selection_pair, single_point_crossover, mutation

# Initialising the variables that will be used for keys and scales
BITS_PER_NOTE = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minor"]
fitness_list = []
pop_id_list = []

# Converts the list of bits into integers
# Returns the number of bits that have made up the song that has been generated
# Parameters:
# @bits = Represents the bits that have been generated for the melody/genome as a List of integers
def int_from_bits(bits: List[int]) -> int:
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))

# Converts the genome into a melody using a Dictionary that saves the notes being played as a string and all the settings of how the melody is played as a list
# Returns melody
# Parameters:
# @genome = Represents the genome that has been generated
# @num_bars = Represents the number of bars (1 bar is made up of 4 bits) that make up the melody/genome
# @num_steps = Represents the number of pitches per note
# @pauses = Represents the pauses/white spaces in the music as an int
# @key = Represents the key (C, C#, Db, etc.) as a string
# @scale = Represents the scale (major, minor) as a string
# @root = Represents the first note in the melody/genome which sets the key that the rest of the melody is generated in as an int
def genome_to_melody(genome: Genome, num_bars: int, num_notes: int, num_steps: int, pauses: int, key: str, scale: str, root: int) -> Dict[str, list]:
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]
    note_length = 4 / float(num_notes)
    scl = EventScale(root=key, scale=scale, first=root)
    
    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }

    for note in notes:
        integer = int_from_bits(note)

        if not pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))

        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]
    
    steps = []
    for step in range(num_steps):
        steps.append([scl[(note+step*2) % len(scl)] for note in melody["notes"]])
    
    melody["notes"] = steps
    return melody

# Converts the genome into the "Events" data type. This is used to specify all the settings of the melody that is being played
# Returns "Events" data type
# Parameters:
# @genome = Represents the genome that has been generated
# @num_bars = Represents the number of bars (1 bar is made up of 4 bits) that make up the melody/genome
# @num_notes = Represents the number of notes that make up the melody/genome
# @num_steps = Represents the number of pitches per note
# @pauses = Represents the pauses/white spaces in the music as a boolean
# @key = Represents the key (C, C#, Db, etc.) as a string
# @scale = Represents the scale (major, minor) as a string
# @root = Represents the first note in the melody/genome which sets the key that the rest of the melody is generated in as an int
# @bpm = Represents the number of beats per minute used to dictate the tempo of the melody as an int
def genome_to_events(genome: Genome, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    return [
        Events(
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.001,
            decay=0.05,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]

# Used to measure the fitness of our generated melody
# Returns rating (of the melody)
# Parameters:
# @genome = Represents the genome that has been generated
# @s = Represents the Server object that is being used
# @num_bars = Represents the number of bars (1 bar is made up of 4 bits) that make up the melody/genome
# @num_notes = Represents the number of notes that make up the melody/genome
# @num_steps = Represents the number of pitches per note
# @pauses = Represents the pauses/white spaces in the music as a boolean
# @key = Represents the key (C, C#, Db, etc.) as a string
# @scale = Represents the scale (major, minor) as a string
# @root = Represents the first note in the melody/genome which sets the key that the rest of the melody is generated in as an int
# @bpm = Represents the number of beats per minute used to dictate the tempo of the melody as an int
def fitness(genome: Genome, s: Server, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, bpm: int) -> int:
    #m = metronome(bpm)

    events = genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
        s.start()

    rating = input("Rating (0 - 10)")

    for e in events:
        e.stop()
    s.stop()
    time.sleep(1)

    try:
        rating = int(rating)
    except ValueError:
        rating = 0
    
    return rating

# Acts as a metronome to keep the tempo of our music
# Returns a tempo
# Parameters:
# @bpm = Represents the number of beats per minute used to dictate the tempo of the melody as an int
def metronome(bpm: int):
    met = Metro(time = 1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(met, table = t, dur  = 0.25, mul = 1)
    freq = Iter(met, choice = [660, 440, 440, 440])
    return Sine(freq = freq, mul = amp).mix(2).out()

# Writes and saves our genome as a MIDI file
# Parameters:
# @filename: Represents the the file that we are saving the genome in
# @genome = Represents the genome that has been generated
# @num_bars = Represents the number of bars (1 bar is made up of 4 bits) that make up the melody/genome
# @num_notes = Represents the number of notes that make up the melody/genome
# @num_steps = Represents the number of pitches per note
# @pauses = Represents the pauses/white spaces in the music as a boolean
# @key = Represents the key (C, C#, Db, etc.) as a string
# @scale = Represents the scale (major, minor) as a string
# @root = Represents the first note in the melody/genome which sets the key that the rest of the melody is generated in as an int
# @bpm = Represents the number of beats per minute used to dictate the tempo of the melody as an int
def save_genome_to_midi(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
        raise ValueError
    
    mf = MIDIFile(1)

    track = 0
    channel = 0

    time = 0.0
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)

    for i, vel in enumerate(melody["velocity"]):
        if vel > 0:
            for step in melody["notes"]:
                mf.addNote(track, channel, step[i], time, melody["beat"][i], vel)

        time += melody["beat"][i]

    os.makedirs(os.path.dirname(filename), exist_ok = True)
    with open(filename, "wb") as f:
        mf.writeFile(f)


@click.command()
@click.option("--num-bars", default=5, prompt='Number of bars:', type=int)
@click.option("--num-notes", default=4, prompt='Notes per bar:', type=int)
@click.option("--num-steps", default=1, prompt='Number of steps:', type=int)
@click.option("--pauses", default=True, prompt='Introduce Pauses?', type=bool)
@click.option("--key", default="C", prompt='Key:', type=click.Choice(KEYS, case_sensitive=False))
@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(SCALES, case_sensitive=False))
@click.option("--root", default=4, prompt='Scale Root:', type=int)
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--num-mutations", default=2, prompt='Number of mutations:', type=int)
@click.option("--mutation-probability", default=0.4, prompt='Mutations probability:', type=float)
@click.option("--bpm", default=128, type=int)

# Our main function
# Returns 0
# @num_bars = Represents the number of bars (1 bar is made up of 4 bits) that make up the melody/genome
# @num_notes = Represents the number of notes that make up the melody/genome
# @num_steps = Represents the number of pitches per note
# @pauses = Represents the pauses/white spaces in the music as a boolean
# @key = Represents the key (C, C#, Db, etc.) as a string
# @scale = Represents the scale (major, minor) as a string
# @root = Represents the first note in the melody/genome which sets the key that the rest of the melody is generated in as an int
# @population_size = Represents the size of the size of the population of melodies that have been generated as an int
# @num_mutations = Represents the number of mutations that have occured as an int (usually seen as the number generations that have passed)
# @mutation_probability = Represents the probability of the mutation effect occurring as a float
# @bpm = Represents the number of beats per minute used to dictate the tempo of the melody as an int
def main(num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, population_size: int, num_mutations: int, mutation_probability: float, bpm: int):
    folder = str(int(datetime.now().timestamp()))   # Sets up the folder

    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]    # Generates a population of genomes

    s = Server().boot() # Starts up the makeshift server

    population_id = 0


    running = True
    while running:
        random.shuffle(population)

        # Sets up the fitness function for the population of genomes
        fitness_func = []
        for genome in population:
            population_fitness = [(genome, fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm))]
            fitness_func.append(population_fitness[0][-1])

        print(f"This is pop fitness {population_fitness[-1]}")
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            # Sets up the next generation of genomes
            parents = selection_pair(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]

        print(f"population {population_id} done")

        time.sleep(1)
        print(f"population fitness is: {fitness_func}")
        average_fitness = (sum(fitness_func) / len(fitness_func))
        print(f"This is the average fitness: {average_fitness}")

        fitness_list.append(average_fitness)        
        print(f"My fitness list: {fitness_list}")

        print("saving population midi …")
        for i, genome in enumerate(population):
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        print("done")

        pop_id_list.append(population_id)

        x = pop_id_list
        y = fitness_list

        plt.title("Evolutionary Fitness Over Multiple Generations")
        plt.plot(x, y, color = "red")
        plt.show()

        running = input("continue? [Y/n]") != "n"
        population = next_generation
        population_id += 1


if __name__ == '__main__':
    main()

# TODO
# IMPLEMENT A GRAPH THAT WILL PLOT THE AVERAGE FITNESS FUNCTION ACROSS MULTIPLE GENERATIONS     √
# TEST DIFFERENT CROSSOVER METHODS (2-POINT CROSSOVER, K-POINT CROSSOVER, UNIFORM CROSSOVER AND ONE OF THE CROSSOVER'S THAT GET USED ON UNORDERED LISTS)
# TEST DIFFERENT FITNESS FUNCTIONS (0 - 5, 0 - 10, 0 - 20)
# TEST DIFFERENT MUTATION RATES (0.1, 0.2, 0.4, 0.6)
# HAVE 10+ DIFFERENT COMBINATIONS OF HYPERPARAMETERS TO WORK WITH FOR VOLUNTEERS
# FIND THESE COMBINATIONS BY WRITING UP A SPREADSHEET WITH THE DIFFERENT COMBINATIONS OF HYPERPARAMETERS