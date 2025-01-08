import random
import math
import certifi
import time
import csv
import re
import time
import functools
import matplotlib.pyplot as plt
import networkx as nx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dataclasses import dataclass, field
from tabulate import tabulate
from typing import List, Tuple


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors')
chrome_options.add_argument(f"--ssl-certificates-path={certifi.where()}")
chrome_options.add_experimental_option(
    "excludeSwitches", ['enable-automation', 'enable-logging'])


@dataclass
class Vertex:
    '''Класс для представления узла графа, который включает название, сокращенное имя и адрес.'''
    name: str
    short_name: str = field(compare=False)
    address: str
    is_visited: bool = field(default=False, repr=False,
                             compare=False, init=False)

    def __str__(self) -> str:
        return self.short_name


class Graph:
    def __init__(self, vertices: List[Vertex]):
        '''Инициализирует граф с заданными узлами и матрицей смежности. '''
        self.vertices = vertices
        self.adjacency_matrix: List[List[int]] = [[1 if i != j else 0 for j in range(
            len(vertices))] for i in range(len(vertices))]

    @property
    def vertices(self) -> List[Vertex]:
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices: List[Vertex]) -> None:
        self.__vertices = vertices

    @property
    def adjacency_matrix(self) -> List[List[int]]:
        return self.__adjacency_matrix

    @adjacency_matrix.setter
    def adjacency_matrix(self, adjacency_matrix: List[List[int]]) -> None:
        self.__adjacency_matrix = adjacency_matrix

    def show_graph(self):
        '''Рисует граф, используя текущую матрицу весов.'''
        G = nx.Graph()
        for i, row in enumerate(self.adjacency_matrix):
            for j, weight in enumerate(row):
                if i < j and (weight != 0 or self.adjacency_matrix[j][i] != 0):
                    G.add_edge(self.vertices[i].short_name, self.vertices[j].short_name, 
                            weight_ab=weight, weight_ba=self.adjacency_matrix[j][i])
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            edge_labels[(u, v)] = f"{d['weight_ab']} / {d['weight_ba']}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", label_pos=0.6)
        plt.show()

    def print_adjacency_matrix(self, show_routes = True) -> None:
        '''Выводит матрицу смежности в консоль.'''
        column_names = [vertex.short_name for vertex in self.vertices]
        table = tabulate(self.adjacency_matrix, headers=column_names,
                         tablefmt='simple', maxcolwidths=3)
        print(table)
        if show_routes:
            for i, vertex_i in enumerate(self.vertices):
                for j, vertex_j in enumerate(self.vertices):
                    if i != j and self.adjacency_matrix[i][j] > 0:
                        print(f'Длина ребра от {vertex_i.short_name} до {vertex_j.short_name}: {self.adjacency_matrix[i][j]}')

    @staticmethod
    def timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            val = func(*args, **kwargs)
            end = time.perf_counter()
            work_time = end - start
            print(f'Время выполнения {func.__name__}: {round(work_time, 4)} сек.')
            return val
        return wrapper

    @timer
    def set_weights(self, gui: bool = True) -> None:
        '''Заполняет матрицу смежности временем достижения между узлами, используя Google Maps.'''
        if not gui:
            chrome_options.add_argument('--headless')
        with webdriver.Chrome(options=chrome_options) as browser:
            url = 'https://www.google.ru/maps'
            browser.get(url)
            route = WebDriverWait(browser, 3).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'hArJGc')))
            route.click()
            time.sleep(0.5)
            k = 1
            for vertex_i in self.vertices:
                for vertex_j in self.vertices:
                    if vertex_i != vertex_j:
                        print(f'\033[91m{vertex_i.address}\033[0m - \033[92m{vertex_j.address}\033[0m')
                        departure_point = WebDriverWait(browser, 10).until(
                            EC.element_to_be_clickable((By.CLASS_NAME, 'tactile-searchbox-input')))
                        departure_point.clear()
                        departure_point.send_keys(vertex_i.address)
                        destination_point = WebDriverWait(browser, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, '[aria-controls="sbsg51"]')))
                        destination_point.clear()
                        destination_point.send_keys(vertex_j.address)
                        destination_point.send_keys(Keys.ENTER)
                        result = WebDriverWait(browser, 10).until(
                            EC.element_to_be_clickable((By.CLASS_NAME, 'Fk3sm'))).text
                        print(f'Маршрут №{
                              k}/{len(self.vertices) ** 2 - len(self.vertices)}: {result}')
                        self.adjacency_matrix[self.vertices.index(
                            vertex_i)][self.vertices.index(vertex_j)] = result
                        k += 1

    def set_weights_from_file(self, filename: str) -> None:
        '''Устанавливает веса из файла с матрицей смежности.'''
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.adjacency_matrix = [[int(i) for i in row] for row in reader]

    def delete_vertex(self, vertex: Vertex) -> None:
        '''Удаляет узел и соответствующие ребра из графа.'''
        try:
            vertex_index = self.vertices.index(vertex)
        except ValueError:
            return
        self.vertices.remove(vertex)
        self.adjacency_matrix = [
            row[:vertex_index] + row[vertex_index+1:] for row in self.adjacency_matrix
        ]
        self.adjacency_matrix = [
            row for i, row in enumerate(self.adjacency_matrix) if i != vertex_index
        ]

    def delete_edge(self, first_vertex: Vertex, second_vertex: Vertex) -> None:
        '''Удаляет ребро между двумя узлами.'''
        try:
            first_index = self.vertices.index(first_vertex)
            second_index = self.vertices.index(second_vertex)
        except ValueError:
            return
        self.adjacency_matrix[first_index][second_index] = 0
        self.adjacency_matrix[second_index][first_index] = 0

    def set_edge(self, first_vertex: Vertex, second_vertex: Vertex, value: int) -> None:
        '''Устанавливает вес ребра между двумя узлами.'''
        try:
            first_vertex_index = self.vertices.index(first_vertex)
            second_vertex_index = self.vertices.index(second_vertex)
        except ValueError:
            return
        self.adjacency_matrix[first_vertex_index][second_vertex_index] = value
        self.adjacency_matrix[second_vertex_index][first_vertex_index] = value

    def add_vertex(self, vertex: Vertex) -> None:
        '''Добавляет новый узел и обновляет матрицу смежности.'''
        self.vertices.append(vertex)
        self.adjacency_matrix.append(
            [1 for _ in range(len(self.vertices) - 1)])
        for i in range(len(self.vertices) - 1):
            self.adjacency_matrix[i].append(1)
        self.adjacency_matrix[len(self.vertices) - 1].append(0)

    def calculate_cost(self, path: List[Vertex]) -> Tuple[int, str]:
        '''Вычисляет стоимость (длину) маршрута для заданного пути.'''
        cost = 0
        calculations = []
        for i in range(len(path) - 1):
            v_from = self.vertices.index(path[i])
            v_to = self.vertices.index(path[i + 1])
            weight = self.adjacency_matrix[v_from][v_to]
            calculations.append(str(weight))
            cost += weight
        return_to_start = self.adjacency_matrix[self.vertices.index(
            path[-1])][self.vertices.index(path[0])]
        calculations.append(str(return_to_start))
        cost += return_to_start
        return cost, " + ".join(calculations) + f" = {cost}"
    
    def normalize_matrix(self):
        '''Нормализует матрицу весов'''
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix)):
                value = str(self.adjacency_matrix[i][j])
                if re.fullmatch(r'\d+ ч \d+ мин.', value):
                    hours = int(re.search(r'\d+ ч', value).group().removesuffix('ч'))
                    minutes = int(re.search(r'\d+ мин.', value).group().removesuffix('мин.'))
                    new_value = hours * 60 + minutes
                elif re.fullmatch(r'\d ч', value):
                    new_value = int(value.removesuffix('ч.')) * 60
                else:
                    new_value = int(value.removesuffix('мин.'))
                self.adjacency_matrix[i][j] = new_value

    def save_matrix_to_csv(self, filename: str) -> None:
        '''Сохраняет матрицу весов в файл.'''
        with open(filename, 'w', newline = '', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in self.adjacency_matrix:
                writer.writerow(row)


class SimulatedAnnealing:
    def __init__(self, graph, k_max: int, T: int | float, alpha: float):
        '''Инициализирует алгоритм имитации отжига.
        Параметры:
            graph (Graph): Граф, на котором будет выполняться алгоритм.
            k_max (int): Максимальное количество итераций.
            T (int): Начальная температура.
            alpha (float): Параметр уменьшения температуры.
        '''
        self.graph = graph
        self.k_max = k_max
        self.T = T
        self.alpha = alpha
        self.current_solution = self.random_solution()
        self.current_cost, _ = self.graph.calculate_cost(self.current_solution)
        self.best_solution = self.current_solution[:]
        self.best_cost = self.current_cost

    def random_solution(self) -> List[Vertex]:
        '''Генерирует случайное начальное решение.'''
        solution = list(self.graph.vertices[1:])
        random.shuffle(solution)
        return [self.graph.vertices[0]] + solution

    def neighbour(self, solution: List[Vertex]) -> List[Vertex]:
        '''Модифицирует текущее решение.'''
        new_solution = solution[1:]
        i, j = random.sample(range(len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return [solution[0]] + new_solution

    def acceptance_probability(self, delta_e: float) -> float:
        '''Вычисляет вероятность принятия нового решения.'''
        return 1.0 if delta_e < 0 else math.exp(-delta_e / self.T)

    def optimize(self) -> Tuple[List[Vertex], int]:
        '''Запускает оптимизацию методом имитации отжига и возвращает лучшее найденное решение.'''
        k = 0
        while self.T > 1e-10 and k < self.k_max:
            new_solution = self.neighbour(self.current_solution)
            new_cost, new_calculation = self.graph.calculate_cost(new_solution)
            delta_e = new_cost - self.current_cost
            print(f"Итерация: {k + 1}/{self.k_max}")
            print(f"Температура: {self.T:.12f}")
            print("Рабочий путь:", ' -> '.join([vertex.short_name for vertex in new_solution]))
            print("Расчёт длины рабочего пути:", new_calculation)
            print(f"Длина рабочего пути: {new_cost}")
            print("Текущий путь:", ' -> '.join([vertex.short_name for vertex in self.current_solution]))
            current_calculation = self.graph.calculate_cost(self.current_solution)[1]
            print("Расчёт длины текущего пути:", current_calculation)
            print(f"Длина текущего пути: {self.current_cost}")
            print(f'Разность энергий: {delta_e}')
            acceptance_probability = self.acceptance_probability(delta_e)
            print(f'Вероятность перехода в новое состояние: {acceptance_probability:.12f}')
            random_num = random.random()
            print(f'Сгенерированное случайное число: {random_num:.12f}')
            print('-' * 40)
            if acceptance_probability > random_num:
                if delta_e >= 0:
                    print('\033[95m' +'Принято худшее решение' + '\033[0m')
                self.current_solution = new_solution
                self.current_cost = new_cost
                if new_cost < self.best_cost:
                    self.best_solution = new_solution
                    self.best_cost = new_cost
            self.T *= self.alpha
            k += 1
        return self.best_solution, self.best_cost


vertex_0 = Vertex('Звезды Арбата',
                  'Отель',
                  'Москва, Новый Арбат, 32')

vertex_1 = Vertex('Московский государственный университет им. М.В. Ломоносова',
                  'МГУ',
                  'Москва, Западный административный округ, район Раменки, территория Ленинские Горы, 1, стр. 52')

vertex_2 = Vertex('Московский государственный технический университет им. Н.Э. Баумана',
                  'МГТУ',
                  'Москва, 2-я Бауманская ул., д. 5, стр. 1')

vertex_3 = Vertex('Московский физико-технический институт',
                  'МФТИ',
                  'Московская область, г. Долгопрудный, Институтский переулок, д. 9.')

vertex_4 = Vertex('Национальный исследовательский ядерный университет «МИФИ»',
                  'МИФИ',
                  'Москва, Каширское шоссе, 31')

vertex_5 = Vertex('Высшая школа экономики',
                  'ВШЭ',
                  'Милютинский переулок, 2/9, Москва, 101000')

vertex_6 = Vertex('Московский государственный институт международных отношений МИД РФ',
                  'МГИМО',
                  'проспект Вернадского, 76кГ, Москва, 119454')

vertex_7 = Vertex('Российская академия народного хозяйства и государственной службы при Президенте РФ',
                  'РАНХиГС',
                  'проспект Вернадского, 84с1, Москва, 119606')

vertex_8 = Vertex('Финансовый университет при Правительстве РФ',
                  'ФУ',
                  'Ленинградский проспект, 51к1, Москва, 125167')

vertex_9 = Vertex('Первый Московский государственный медицинский университет им. И.М. Сеченова',
                  'МГМУ',
                  'Трубецкая улица, 8с2, Москва, 119048')

vertex_10 = Vertex('Российский экономический университет им. Г.В. Плеханова',
                   'РЭУ',
                   'Стремянный переулок, 36, Москва, 115054')

vertex_11 = Vertex('Университет науки и технологий МИСИС',
                   'МИСИС',
                   'Ленинский проспект, 2/4, Москва, 119049')

vertex_12 = Vertex('Российский университет дружбы народов',
                   'РУДН',
                   'улица Миклухо-Маклая, 6, Москва, 117198')

vertex_13 = Vertex('Российский национальный исследовательский медицинский университет им. Н.И. Пирогова',
                   'РНИМУ',
                   'улица Островитянова, 1с7, Москва, 117513')

vertex_14 = Vertex('Московский авиационный институт',
                   'МАИ',
                   'Волоколамское шоссе, 4к6, Москва, 125310')

vertex_15 = Vertex('Национальный исследовательский университет «МЭИ»',
                   'МЭИ',
                   'Красноказарменная ул., 17 строение 1Г, Москва, 111250')

vertex_16 = Vertex('Московский государственный юридический университет им. О.Е. Кутафина',
                   'МГЮА',
                   'Садовая-Кудринская улица, 9с1, Москва, 123242')

vertex_17 = Vertex('Российский государственный университет нефти и газа им. И. М. Губкина',
                   'РГУ',
                   'Ленинский проспект, 65к1, Москва, 119296')

vertex_18 = Vertex('Московский педагогический государственный университет',
                   'МПГУ',
                   'проспект Вернадского, 88, Москва, 119571')

vertex_19 = Vertex('Национальный исследовательский Московский государственный строительный университет',
                   'НИУ МГСУ',
                   'Ярославское шоссе, 26к1, Москва, 129337')

vertex_20 = Vertex('Московский государственный лингвистический университет',
                   'МГЛУ',
                   'улица Остоженка, 38с1, Москва, 119034')

vertex_21 = Vertex('Всероссийская академия внешней торговли',
                   'ВАВТ',
                   'Воробьёвское шоссе, 6А, Москва, 119285')

vertex_22 = Vertex('Российский химико-технологический университет им. Д.И. Менделеева',
                   'РХТУ',
                   'Миусская площадь, 9, Москва')

vertex_23 = Vertex('МИРЭА – Российский технологический университет',
                   'МИРЭА',
                   'проспект Вернадского, 86с2, Москва')

vertices = [vertex_0, vertex_1, vertex_2, vertex_3, vertex_4,
            vertex_5, vertex_6, vertex_7, vertex_8, vertex_9,
            vertex_10, vertex_11, vertex_12, vertex_13, vertex_14,
            vertex_15, vertex_16, vertex_17, vertex_18, vertex_19,
            vertex_20, vertex_21, vertex_22, vertex_23]

graph = Graph(vertices)
# graph.set_weights()
# graph.print_adjacency_matrix(False)
# graph.normalize_matrix()
graph.set_weights_from_file('universities_info.csv')
# graph.print_adjacency_matrix(False)
# graph.show_graph()

solution = SimulatedAnnealing(graph, 100, 100, 0.5)
best_solution, best_cost = solution.optimize()
print("Лучший найденный путь:", ' -> '.join([vertex.short_name for vertex in best_solution]))
print("Время пути:", best_cost)
print("Текущий путь: ", ' -> '.join([vertex.short_name for vertex in solution.current_solution]))
print("Время пути:", solution.current_cost)


# test_graph = Graph([vertex_0, vertex_2, vertex_3, vertex_4, vertex_7, vertex_11, vertex_23])
# # test_graph.set_weights()
# # test_graph.print_adjacency_matrix(False)
# # test_graph.normalize_matrix()
# # test_graph.save_matrix_to_csv('shit.csv')
# test_graph.set_weights_from_file('universities_test.csv')
# test_graph.print_adjacency_matrix()
# test_graph.show_graph()

# sol = SimulatedAnnealing(test_graph, 100, 100, 0.5)
# best_solution, best_cost = sol.optimize()
# print("Лучший найденный путь:", ' -> '.join([vertex.short_name for vertex in best_solution]))
# print("Время пути:", best_cost)
# print("Текущий путь: ", ' -> '.join([vertex.short_name for vertex in sol.current_solution]))
# print("Время пути:", sol.current_cost)
