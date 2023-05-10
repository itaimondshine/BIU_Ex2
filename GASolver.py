from random import randint, uniform
from random import sample
import numpy as np


class GeneticSolver:
    def __init__(self):
        # Genetic Algorithm Parameters
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
            probability = self.tournament_probabilities[i-1] * (1.0 - self.tournament_winner_probability)
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

    def calculate_key_fitness(self, text):
        unigrams = self.generate_ngrams(text, 1)
        bigrams = self.generate_ngrams(text, 2)
        trigrams = self.generate_ngrams(text, 3)

        bigram_fitness = sum([self.bigram_frequency[bigram] for bigram in bigrams if
                              bigram in self.bigram_frequency and self.bigram_weight > 0])

        words = text.lower().split()
        words_appear = len(set(self.list_of_words).intersection(words))

        trigram_fitness = sum([self.trigram_frequency[trigram] for trigram in trigrams if
                               trigram in self.trigram_frequency and self.trigram_weight > 0])

        unigrams_fitness = sum([self.trigram_frequency[unigram] for unigram in unigrams if
                                unigram in self.trigram_frequency and self.unigram_weight > 0])

        fitness = (bigram_fitness * self.bigram_weight) + (trigram_fitness * self.trigram_weight) + (
                    unigrams_fitness * self.unigram_weight) + words_appear

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
        crossover_population = []

        while len(crossover_population) < self.crossover_count:
            parent_one, parent_two = None, None

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
            r = uniform(0, 1)

            if r < self.mutation_probability:
                key = population[i]
                mutated_key = self.mutate_key(key)

                population[i] = mutated_key

        return population


    def convert_to_plaintext(self, decrypted_text):
        plaintext = [c.lower() if self.lettercase[i] else c for i, c in enumerate(decrypted_text)]
        return ''.join(plaintext)


    def get_list_of_words(self, filename):
        with open(filename, 'r') as f:
            contnet = f.read()

        return contnet.split('\n')


    def solve(self, ciphertext = ''):
        # Defining ciphertext
        self.ciphertext = ciphertext

        # Checking if ciphertext is valid
        if self.ciphertext == '':
            message = (
                '\n(GeneticSolver) Ciphertext invalid. Use solve() as such:\n'
                '\tsolver = GeneticSolver()\n'
                '\tsolver.solve("Example ciphertext")'
            )
            print(message)
            return

        # Formatting ciphertext
        self.lettercase = [ch.islower() and ch.isalpha() for ch in self.ciphertext]
        self.ciphertext = self.ciphertext.upper()

        # Getting pre-computed ngram freqread_letter_frequenciesuency
        bigram_filename = 'data/Letter2_Freq.txt'
        self.bigram_frequency = self.read_letter_frequencies(bigram_filename)

        unigram_filename = 'data/Letter_Freq.txt'
        self.unigram_frequency =  self.read_letter_frequencies(unigram_filename)

        trigram_filename = 'data/dict.txt'
        self.trigram_frequency =  self.get_char_trigram_dict(trigram_filename)

        dict_filename = 'data/dict.txt'
        self.list_of_words =  self.get_list_of_words(dict_filename)

        # Main Program
        population = self.initialization()

        highest_fitness = 0
        stuck_counter = 0
        for no in range(self.generations + 1):
            fitness = self.evaluation(population)
            elitist_population = self.elitism(population, fitness)
            crossover_population = self.reproduction(population, fitness)

            population = elitist_population + crossover_population

            # Terminate if highest_fitness not increasing
            if highest_fitness == max(fitness):
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter >= self.terminate:
                break

            highest_fitness = max(fitness)
            average_fitness = sum(fitness) / self.population_size

            index = fitness.index(highest_fitness)
            key = population[index]
            decrypted_text = self.decrypt(key)

            if self.verbose:
                plaintext = self.convert_to_plaintext(decrypted_text)

                print('[Generation ' + str(no) + ']',)
                print('Average Fitness:', average_fitness)
                print('Max Fitness:', highest_fitness)
                print('Key:', key)
                print('Decrypted Text:\n' + plaintext + '\n')

        plaintext = self.convert_to_plaintext(decrypted_text)
        return plaintext


