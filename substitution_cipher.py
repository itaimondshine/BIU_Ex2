from random import randint, sample
from enum import Enum
import numpy as np
from consts import *
import random


class SolverType(Enum):
    REGULAR = 0,
    DARWIN = 1,
    LAMARCK = 2


# --------------------------------------Util Functions -----------------------------#


def save_plain_to_file(plain_text):
    with open(PLAIN_FILE_PATH, 'w') as pf:
        pf.write(plain_text)


def save_perm_to_file(key: str):
    with open(PERM_FILE_PATH, 'w') as f:
        for i, char in enumerate(key):
            f.write('{} {}\n'.format(chr(i + 65), char))


def get_best_results(population, fitness):
    highest_fitness = max(fitness)
    index = fitness.index(highest_fitness)
    key = population[index]

    return highest_fitness, key


def get_list_of_words(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content.split('\n')


def read_letter_frequencies(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    frequencies = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts[1:]) > 0:
            letter = str(parts[1:][0])
            frequency = float(parts[0])
            frequencies[letter] = frequency

    return frequencies


def get_char_bigram_dict(file_name):
    """
    Get a dict of mapping between char bigram and number of appreances in the text.

    Args:
      file_name: The name of the file to read.

    Returns:
      A dict of mapping between char bigram and number of appreances in the text.
    """

    with open(file_name, 'r') as f:
        words = f.read().replace('\n', '')

    char_bigram_dict = {}
    for i in range(len(words) - 1):
        char_bigram = words[i] + words[i + 1]
        if char_bigram.isalpha():
            if char_bigram not in char_bigram_dict:
                char_bigram_dict[char_bigram] = 1
            else:
                char_bigram_dict[char_bigram] += 1

    return char_bigram_dict


def generate_ngrams(word, n):
    ngrams = [word[i:i + n] for i in range(len(word) - n + 1)]
    return ngrams


def calculate_ngram_fitness(ngrams, ngram_frequency, ngram_weight):
    return sum([ngram_frequency[ngram] * ngram_weight for ngram in ngrams if
                ngram in ngram_frequency and ngram_weight > 0])


def mutate_key(key):
    a, b = randint(0, len(key) - 1), randint(0, len(key) - 1)
    key = list(key)
    key[a], key[b] = key[b], key[a]
    return ''.join(key)


def get_char_trigram_dict(file_name):

    with open(file_name, 'r') as f:
        words = f.read().replace('\n', '')

    char_trigram_dict = {}
    for i in range(len(words) - 2):
        char_trigram = words[i] + words[i + 1] + words[i + 2]
        if char_trigram.isalpha():
            if char_trigram not in char_trigram_dict:
                char_trigram_dict[char_trigram.upper()] = 1
            else:
                char_trigram_dict[char_trigram.upper()] += 1

    return char_trigram_dict


class GeneticSubstitutionSolver:
    def __init__(self, ciphertext):
        self.ciphertext = ciphertext
        self.lettercase = [ch.islower() and ch.isalpha() for ch in self.ciphertext]
        self.ciphertext = self.ciphertext.upper()
        self.bigram_frequency = read_letter_frequencies(BIGRAM_FILENAME_PATH)
        self.unigram_frequency = read_letter_frequencies(UNIGRAM_FILENAME_PATH)
        self.set_of_words = set(get_list_of_words(DICT_FILENAME_PATH))
        self.set_of_words_upper = {word.upper() for word in self.set_of_words}
        self.trigram_frequency = get_char_trigram_dict(DICT_FILENAME_PATH)
        self.generations = 500
        self.population_size = 500
        self.tournament_size = 20
        self.tournament_winner_probability = 0.75
        self.crossover_probability = 0.65
        self.crossover_points_count = 5
        self.mutation_probability = 0.2
        self.elitism_percentage = 0.15
        self.selection_method = 'TS'
        self.terminate = 100

        # Other parameters
        self.bigram_weight = 0.4
        self.unigram_weight = 0
        self.trigram_weight = 0.6
        # Usage parameters
        self.verbose = False

        # Default variables
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.elitism_count = int(self.elitism_percentage * self.population_size)
        self.crossover_count = self.population_size - self.elitism_count

        self.tournament_probabilities = [self.tournament_winner_probability]

        for i in range(1, self.tournament_size):
            probability = self.tournament_probabilities[i - 1] * (1.0 - self.tournament_winner_probability)
            self.tournament_probabilities.append(probability)

    def info_display(self, decrypted_text, highest_fitness, key, no):
        plaintext = self.convert_to_plaintext(decrypted_text)
        print('[Generation ' + str(no) + ']', )
        print('Max Fitness:', highest_fitness)
        print('Key:', key)
        print('Decrypted Text:\n' + plaintext + '\n')

    def decrypt(self, key):
        letter_mapping = {self.letters[i]: key.upper()[i] for i in range(26)}

        decrypted_text = ''
        for character in self.ciphertext:
            decrypted_text += letter_mapping.get(character, character)

        return decrypted_text

    def calculate_key_fitness(self, text):

        unigrams = generate_ngrams(text, 1) if self.unigram_weight else []
        bigrams = generate_ngrams(text, 2)
        trigrams = []  # self.generate_ngrams(text, 3)

        unigrams_fitness = calculate_ngram_fitness(unigrams, self.unigram_frequency,
                                                   self.unigram_weight) if self.unigram_weight else 0
        bigrams_fitness = calculate_ngram_fitness(bigrams, self.bigram_frequency, self.bigram_weight)

        trigrams_fitness = 0  # self.calculate_ngram_fitness(trigrams, self.trigram_frequency, self.trigram_weight)

        words = text.split()
        words_appear = len(self.set_of_words_upper.intersection(words))

        fitness = (unigrams_fitness + bigrams_fitness + trigrams_fitness) + words_appear

        return fitness

    def combine_strings(self, string1, string2):
        crossover_points = sample(range(26), self.crossover_points_count)
        offspring = [string1[i] if i in crossover_points else None for i in range(26)]

        for char in string2:
            if char not in offspring:
                offspring[offspring.index(None)] = char

        return ''.join(offspring)

    def initialization(self):
        population = []
        for _ in range(self.population_size):
            key = ''.join(sample(self.letters, len(self.letters)))
            population.append(key)
        return population

    def evaluation(self, population):
        return [self.calculate_key_fitness(self.decrypt(key)) for key in population]

    def elitism(self, population, fitness):
        population_fitness = {key: value for key, value in zip(population, fitness)}
        sorted_population = [key for key, value in sorted(population_fitness.items(), key=lambda item: item[1])]

        elitist_population = sorted_population[-self.elitism_count:]

        return elitist_population

    def select_by_roulette(self, fitness):
        probabilities = np.array(fitness) / sum(fitness)
        return np.random.choice(range(self.population_size), p=probabilities)

    def select_by_tournament(self, population, fitness):
        tournament_indices = np.random.choice(len(population), size=(2, self.tournament_size), replace=False)
        tournament_fitness = np.take(fitness, tournament_indices)
        tournament_keys = np.take(population, tournament_indices)
        sorted_indices = np.argsort(tournament_fitness, axis=1)[:, ::-1]
        winner_indices = np.random.choice(self.tournament_size, size=2, p=self.tournament_probabilities)
        return np.array([tournament_keys[i][sorted_indices[i][winner_indices[i]]] for i in range(2)])

    def generate_offspring(self, parent_one, parent_two):
        offspring_one = self.combine_strings(parent_one, parent_two)
        offspring_two = self.combine_strings(parent_two, parent_one)
        return offspring_one, offspring_two

    def reproduction(self, pop, fit):

        crossover_population = []

        while len(crossover_population) < self.crossover_count:

            if self.selection_method == 'RWS':
                parent_one_index = self.select_by_tournament(pop, fit)
                parent_two_index = self.select_by_roulette(fit)

                parent_one = pop[parent_one_index]
                parent_two = pop[parent_two_index]
            else:
                parent_one, parent_two = self.select_by_tournament(pop, fit)

            crossover_population += [*self.generate_offspring(parent_one, parent_two)]

        crossover_population = self.mutation(crossover_population, self.crossover_count)

        return crossover_population

    def mutation(self, population, population_size):

        for i in range(population_size):
            r = random.random()
            if r < self.mutation_probability:
                population[i] = mutate_key(population[i])

        return population

    def convert_to_plaintext(self, decrypted_text):
        plaintext = [c.lower() if self.lettercase[i] else c for i, c in enumerate(decrypted_text)]
        return ''.join(plaintext)

    def new_gen(self, population):
        fitness = self.evaluation(population)
        elitist_population = self.elitism(population, fitness)
        crossover_population = self.reproduction(population, fitness)
        population = elitist_population + crossover_population
        return population, fitness

    def optimize(self, population, fitness):
        texts = [self.decrypt(key) for key in population]
        best_index = fitness.index(max(fitness))
        best_text_str = texts[best_index].upper()
        best_text = best_text_str.split()
        best_key = population[best_index]
        best_key_dict = {letter: i for (i, letter) in enumerate(best_key)}
        max_fitness = max(fitness)
        for word in best_text:
            if len(word) < 5 or word in self.set_of_words_upper or (set(word) - UPPER_LETTERS):
                continue
            last_letter = word[-1]
            if last_letter not in UPPER_LETTERS:
                continue
            best_template = best_text_str.replace(last_letter, '$')
            for letter in UPPER_LETTERS:
                new_text = best_template.replace(letter, last_letter).replace('$', letter)
                if self.calculate_key_fitness(new_text) > max_fitness:
                    iter_best_key = list(best_key)
                    iter_best_key[best_key_dict[last_letter]] = letter
                    iter_best_key[best_key_dict[letter]] = last_letter
                    iter_best_key = "".join(iter_best_key)
                    population[best_index] = iter_best_key
                    print("Optimized")
                    return population

        return population

    def solve(self, solver=SolverType.REGULAR):
        population = self.initialization()

        key = ''
        plaintext = ''
        highest_fitness = 0
        convergence_counter = 0
        for gen in range(self.generations + 1):

            population, fitness = self.new_gen(population)
            if solver != SolverType.REGULAR:
                optimized_population = self.optimize(population, fitness)
                fitness = self.evaluation(optimized_population)

            if solver == SolverType.LAMARCK:
                population = optimized_population
            # Terminate if highest_fitness not increasing
            if highest_fitness == max(fitness):
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter >= self.terminate:
                break

            highest_fitness, key = get_best_results(population, fitness)

            plaintext = self.convert_to_plaintext(self.decrypt(key))

            self.info_display(plaintext, highest_fitness, key, gen)

        save_plain_to_file(plain_text=plaintext)
        save_perm_to_file(key=key)

        return plaintext
