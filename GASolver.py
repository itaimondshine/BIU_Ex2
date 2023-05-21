from random import randint, uniform
from random import sample
from copy import deepcopy
from enum import Enum
import numpy as np
import settings
import random
import string

LETTERS = set(string.ascii_lowercase)

class SolverType(Enum):
    REGULAR = 0,
    DARWIN = 1,
    LAMARCK = 2
class GeneticSolver:
    def __init__(self, ciphertext):
        # Genetic Algorithm Parameters
        self.ciphertext = ciphertext
        self.lettercase = [ch.islower() and ch.isalpha() for ch in self.ciphertext]
        self.ciphertext = self.ciphertext.upper()
        self.bigram_frequency = self.read_letter_frequencies(settings.BIGRAM_FILENAME_PATH)
        self.unigram_frequency = self.read_letter_frequencies(settings.UNIGRAM_FILENAME_PATH)
        self.list_of_words = self.get_list_of_words(settings.DICT_FILENAME_PATH)
        self.set_of_words = set(self.list_of_words)
        self.trigram_frequency = self.get_char_trigram_dict(settings.DICT_FILENAME_PATH)
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

    def get_char_trigram_dict(self, file_name):
        """
        Get a dict of mapping between char trigram and number of appreances in the text.

        Args:
          file_name: The name of the file to read.

        Returns:
          A dict of mapping between char trigram and number of appreances in the text.
        """

        with open(file_name, 'r') as f:
            words = f.read().replace('\n', '')

        char_trigram_dict = {}
        for i in range(len(words) - 2):
            char_trigram = words[i] + words[i + 1] + words[i + 2]
            if char_trigram.isalpha():
                if char_trigram not in char_trigram_dict:
                    char_trigram_dict[char_trigram] = 1
                else:
                    char_trigram_dict[char_trigram] += 1

        return char_trigram_dict

    def read_letter_frequencies(self, filename):
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

    def get_char_bigram_dict(self, file_name):
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

    def generate_ngrams(self, word, n):
        ngrams = [word[i:i + n] for i in range(len(word) - n + 1)]
        processed_ngrams = [ngram.upper() for ngram in ngrams if ngram.isalpha()]
        return processed_ngrams

    def decrypt(self, key):
        letter_mapping = {self.letters[i]: key.upper()[i] for i in range(26)}

        decrypted_text = ''
        for character in self.ciphertext:
            decrypted_text += letter_mapping.get(character, character)

        return decrypted_text

    def calculate_ngram_fitness(self, ngrams, ngram_frequency, ngram_weight):
        """
        This function calculates the fitness of a text based on the frequency of ngrams.

        Args:
          ngrams: The ngrams in the text.
          ngram_frequency: The frequency of each ngram.
          ngram_weight: The weight of each ngram.

        Returns:
          The fitness of the text.
        """

        return sum([ngram_frequency[ngram] * ngram_weight for ngram in ngrams if
                    ngram in ngram_frequency and ngram_weight > 0])

    def calculate_key_fitness(self, text):
        """
        This function calculates the fitness of a text based on its unigrams, bigrams, and trigrams.

        Args:
          text: The text to calculate the fitness of.

        Returns:
          The fitness of the text.
        """

        unigrams = self.generate_ngrams(text, 1)
        bigrams = self.generate_ngrams(text, 2)
        trigrams = self.generate_ngrams(text, 3)

        unigrams_fitness = self.calculate_ngram_fitness(unigrams, self.unigram_frequency, self.unigram_weight)
        bigrams_fitness = self.calculate_ngram_fitness(bigrams, self.bigram_frequency, self.bigram_weight)
        trigrams_fitness = self.calculate_ngram_fitness(trigrams, self.trigram_frequency, self.trigram_weight)

        words = text.lower().split()
        words_appear = len(set(self.list_of_words).intersection(words))

        fitness = (unigrams_fitness + bigrams_fitness + trigrams_fitness) + words_appear

        return fitness

    def merge_keys(self, one, two):
        crossover_points = sample(range(26), self.crossover_points_count)
        offspring = [one[i] if i in crossover_points else None for i in range(26)]

        for ch in two:
            if ch not in offspring:
                offspring[offspring.index(None)] = ch

        return ''.join(offspring)

    def mutate_key(self, key):
        a, b = randint(0, len(key) - 1), randint(0, len(key) - 1)
        key = list(key)
        key[a], key[b] = key[b], key[a]
        return ''.join(key)

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

    def roulette_wheel_selection(self, fitness):
        probabilities = np.array(fitness) / sum(fitness)
        index = np.random.choice(range(self.population_size), p=probabilities)
        return index

    def tournament_selection(self, population, fitness):
        selected_keys = []
        for i in range(2):
            tournament_indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
            tournament_fitness = [fitness[j] for j in tournament_indices]
            tournament_keys = [population[j] for j in tournament_indices]
            sorted_indices = np.argsort(tournament_fitness)[::-1]
            winner_index = np.random.choice(self.tournament_size, p=self.tournament_probabilities)
            selected_keys.append(tournament_keys[sorted_indices[winner_index]])
        return selected_keys

    def reproduction(self, population, fitness):
        """
        This function reproduces the population based on the fitness of each individual.

        Args:
          population: The current population.
          fitness: The fitness of each individual in the population.

        Returns:
          The new population.
        """

        crossover_population = []

        while len(crossover_population) < self.crossover_count:

            if self.selection_method == 'RWS':
                parent_one_index = self.roulette_wheel_selection(fitness)
                parent_two_index = self.roulette_wheel_selection(fitness)

                parent_one = population[parent_one_index]
                parent_two = population[parent_two_index]
            else:
                parent_one, parent_two = self.tournament_selection(population, fitness)

            offspring_one = self.merge_keys(parent_one, parent_two)
            offspring_two = self.merge_keys(parent_two, parent_one)

            crossover_population += [offspring_one, offspring_two]

        crossover_population = self.mutation(crossover_population, self.crossover_count)

        return crossover_population

    def mutation(self, population, population_size):

        for i in range(population_size):
            r = random.random()
            if r < self.mutation_probability:
                population[i] = self.mutate_key(population[i])

        return population

    def convert_to_plaintext(self, decrypted_text):
        plaintext = [c.lower() if self.lettercase[i] else c for i, c in enumerate(decrypted_text)]
        return ''.join(plaintext)

    def get_list_of_words(self, filename):
        with open(filename, 'r') as f:
            contnet = f.read()
        return contnet.split('\n')

    def new_gen(self, population):
        fitness = self.evaluation(population)
        elitist_population = self.elitism(population, fitness)
        crossover_population = self.reproduction(population, fitness)
        population = elitist_population + crossover_population
        return population, fitness

    def get_best_results(self, population, fitness):
        highest_fitness = max(fitness)
        index = fitness.index(highest_fitness)
        key = population[index]

        return highest_fitness, key

    def optimize(self, population, fitness):
        texts = [self.decrypt(key).lower().split() for key in population]
        best_index = fitness.index(max(fitness))
        best_text = texts[best_index]
        best_key = population[best_index].lower()
        best_key_dict = {letter : i for (i, letter) in enumerate(best_key)}
        #best_text_copy = deepcopy(best_text)

        for word in best_text:
            if len(word) > 3 and word in self.set_of_words:
                continue
            last_letter = word[-1]
            if last_letter not in string.ascii_lowercase:
                continue
            for letter in LETTERS:
                iter_best_key = list(deepcopy(best_key))
                iter_best_key[best_key_dict[last_letter]] = letter
                iter_best_key[best_key_dict[letter]] = last_letter
                iter_best_key = "".join(iter_best_key)
                if self.calculate_key_fitness(self.decrypt(iter_best_key)) > max(fitness):
                    # new_population = []
                    # for key in population:
                    #     key.index(letter), key.index_f
                    #     new_population.append(key[]
                    population[best_index] = iter_best_key.upper()
                    return population


        return population

        a = 2
    def solve(self, solver=SolverType.REGULAR):
        # Main Program
        population = self.initialization()

        plaintext = ''
        highest_fitness = 0
        stuck_counter = 0
        for no in range(self.generations + 1):

            population, fitness = self.new_gen(population)
            if solver != SolverType.REGULAR:
                optimized_population = self.optimize(population, fitness)
                fitness = self.evaluation(optimized_population)

            if solver == SolverType.LAMARCK:
                population = optimized_population
            # Terminate if highest_fitness not increasing
            if highest_fitness == max(fitness):
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter >= self.terminate:
                break

            highest_fitness, key = self.get_best_results(population, fitness)

            plaintext = self.convert_to_plaintext(self.decrypt(key))

            if self.verbose:
                self.verbose_display(sum(fitness) / self.population_size, plaintext, highest_fitness, key, no)

        return plaintext

    def verbose_display(self, average_fitness, decrypted_text, highest_fitness, key, no):
        plaintext = self.convert_to_plaintext(decrypted_text)
        print('[Generation ' + str(no) + ']', )
        print('Average Fitness:', average_fitness)
        print('Max Fitness:', highest_fitness)
        print('Key:', key)
        print('Decrypted Text:\n' + plaintext + '\n')
