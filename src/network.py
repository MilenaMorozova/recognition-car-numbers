import random
import json
import os

import cv2
import numpy as np

possible_values = [str(i) for i in range(10)]+['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']
characters = {key: i for i, key in enumerate(possible_values)}


class Network(object):

    def __init__(self, sizes, filename):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.evaluation = 0
        self.parameters_filename = filename

    def feedforward(self, input_data):
        """
        Output recognition result
        :param input_data: np.array image array in one string
        :return: recognition result
        """
        for b, w in zip(self.biases, self.weights):
            summa = np.dot(w, input_data) + b
            input_data = self.sigmoid(summa)
        return input_data

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """We train the network using mini-packages and stochastic gradient descent.
        :param training_data: list[tuple]
                tuple(image_array, answer)
                image_array - np.array array of image pix in one string
        :param epochs:
        :param mini_batch_size:
        :param eta: learning speed
        :param test_data: list[tuple]
                tuple(image_array, answer)
                image_array - np.array
                If test_data exists, then the network will be evaluated against the verification data after each era,
         and the current progress will be displayed.
        :return: nothing
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_test = len(test_data)
                eval = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}".format(j, eval, n_test))
                if eval > self.evaluation:
                    self.evaluation = eval
                    self.save_weights(self.parameters_filename)
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Обновить веса и смещения сети, применяя градиентный спуск
        с использованием обратного распространения к одному мини-пакету.
        mini_batch – это список кортежей (x, y), а eta – скорость обучения."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.
        ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямой проход
        activation = x
        activations = [x]  # список для послойного хранения активаций
        zs = []  # список для послойного хранения z-векторов
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # обратный проход
        # delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Вернуть количество проверочных входных данных, для которых нейросеть выдаёт правильный результат.
        Выходные данные сети – это номер нейрона в последнем слое с наивысшим уровнем активации."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Вернуть вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return (output_activations - y)

    # Разные функции
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def save_weights(self, file_name: str):
        with open(file_name, 'w') as file:
            json.dump({'evaluation': self.evaluation,
                       'weights': [i.tolist() for i in self.weights],
                       'biases': [i.tolist() for i in self.biases]}, file)

    def load_weights(self, file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
            self.evaluation = data.get('evaluation')
            self.weights = list(map(np.array, data.get('weights')))
            self.biases = list(map(np.array, data.get('biases')))


def create_training_data(directory_name: list) -> list:
    """
    :param directory_name: list - path to files for training
    :return: list[tuple] - tuple:(matrix of image, answer)
            matrix of image - np.array
            answer - zero-vector with 1 on i-place
    """
    training_data = []
    for directory in directory_name:
        images = os.listdir(path=directory)
        for image in images:
            char_code_name = characters[image.split('__')[0]]
            image_array = np.array([np.mean(pix)/255. for row in cv2.imread(os.path.join(directory, image)) for pix in row])

            answer = np.zeros((22, 1))
            answer[char_code_name] = 1.

            training_data.append((np.reshape(image_array, (972, 1)),  np.reshape(answer, (22, 1))))

    print('Training data is complete')
    random.shuffle(training_data)
    return training_data


# if __name__ == '__main__':
#     training_data = create_training_data(['..\\training_data\\letters', '..\\training_data\\digits'])
#     test_data = create_training_data(['..\\test_data\\letters', '..\\test_data\\digits'])
#
#     net = Network([972, 250, 100, 22], 'network_parameters.json')
#     net.SGD(training_data, 10, 220, 1.3, test_data)
#     # test_data = training_data
#
#     # net.load_weights('network_parameters.json')
#     print('Correct - Incorrect')
#     numbers = 0
#     for test in test_data:
#         answer = np.argmax(test[1])
#         result = np.argmax(net.feedforward(test[0]))
#         # print(possible_values[answer], '-', possible_values[result])
#         if answer == result:
#             numbers += 1
#         else:
#             print(possible_values[answer], '-', possible_values[result])
#     print('TOTAL:', numbers, '/', len(test_data))
