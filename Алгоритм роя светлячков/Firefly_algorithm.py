import random
import math
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def goldstein_price(x: float, y: float) -> float:
    '''
    Функция Голдштейна-Прайса для оптимизации.
    Параметры:
        x (float): Координата по оси x.
        y (float): Координата по оси y.
    Возвращает:
        float: Значение функции Голдштейна-Прайса.
    '''
    term1 = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 *
             x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
    term2 = (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 *
             x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return term1 * term2


class Firefly:
    def __init__(self, position: List[float], luciferin: float = 0.0, radius: float = 1.0):
        '''
        Инициализирует светлячка.
        Параметры:
            position (List[float]): Начальная позиция светлячка.
            luciferin (float): Уровень люциферина.
            radius (float): Радиус окрестности светлячка.
        '''
        self.position = position
        self.luciferin = luciferin
        self.radius = radius

    def update_luciferin(self, function_value: float, rho: float, gamma: float):
        '''
        Обновляет уровень люциферина светлячка.
        Параметры:
            function_value (float): Значение целевой функции в текущей позиции.
            rho (float): Коэффициент уменьшения люциферина.
            gamma (float): Коэффициент привлекательности светлячка.
        '''
        self.luciferin = (1 - rho) * self.luciferin + \
            gamma * (1 / function_value)

    def move_towards(self, other: 'Firefly', delta: float, bounds: List[Tuple[float, float]]):
        '''
        Перемещает светлячка в направлении другого более яркого светлячка.
        Параметры:
            other (Firefly): Другой светлячок, к которому перемещается текущий.
            delta (float): Коэффициент изменения позиции.
            bounds (List[Tuple[float, float]]): Границы пространства поиска.
        '''
        direction = [other.position[i] - self.position[i]
                     for i in range(len(self.position))]
        distance = math.sqrt(sum(d ** 2 for d in direction))
        if distance > 0:
            normalized_direction = [d / distance for d in direction]
            self.position = [
                min(max(
                    self.position[i] + delta * normalized_direction[i], bounds[i][0]), bounds[i][1])
                for i in range(len(self.position))
            ]


class FireflySwarm:
    def __init__(self, fitness_function: Callable[..., float], bounds: List[Tuple[float, float]],
                 num_fireflies: int, max_iterations: int, beta: float, rho: float,
                 delta: float, gamma: float, initial_radius: float):
        '''
        Инициализирует алгоритм роя светлячков.
        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            bounds (List[Tuple[float, float]]): Границы пространства поиска [(xmin, xmax), (ymin, ymax)].
            num_fireflies (int): Количество светлячков.
            max_iterations (int): Максимальное количество итераций.
            beta (float): Коэффициент изменения радиуса окрестности.
            rho (float): Коэффициент уменьшения уровня люциферина.
            delta (float): Коэффициент изменения позиции.
            gamma (float): Коэффициент увеличения люциферина.
            initial_radius (float): Начальный радиус окрестности.
        '''
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.num_fireflies = num_fireflies
        self.max_iterations = max_iterations
        self.beta = beta
        self.rho = rho
        self.delta = delta
        self.gamma = gamma
        self.initial_radius = initial_radius
        self.fireflies = []
        self.min_radius = 0.1
        self.max_radius = max(
            bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])
        self.best_values = []
        self.positions_history = []

    def initialize_fireflies(self):
        '''Инициализирует популяцию светлячков в случайных позициях.'''
        self.fireflies = [
            Firefly(
                position=[random.uniform(bounds[0], bounds[1])
                          for bounds in self.bounds],
                radius=self.initial_radius
            )
            for _ in range(self.num_fireflies)
        ]

    def calculate_neighbors(self, firefly: Firefly) -> List[Firefly]:
        '''
        Вычисляет множество соседей светлячка.
        Параметры:
            firefly (Firefly): Текущий светлячок.
        Возвращает:
            List[Firefly]: Список соседей светлячка.
        '''
        return [
            other for other in self.fireflies
            if other is not firefly
            and math.dist(firefly.position, other.position) < firefly.radius
            and firefly.luciferin < other.luciferin
        ]

    def calculate_probabilities(self, firefly: Firefly, neighbors: List[Firefly]) -> List[float]:
        '''
        Вычисляет вероятность перемещения к соседям.
        Параметры:
            firefly (Firefly): Текущий светлячок.
            neighbors (List[Firefly]): Список соседей.
        Возвращает:
            List[float]: Список вероятностей перемещения к каждому соседу.
        '''
        total_difference = sum(
            other.luciferin - firefly.luciferin for other in neighbors)
        if total_difference == 0:
            return [1 / len(neighbors)] * len(neighbors)
        probabilities = [(other.luciferin - firefly.luciferin) /
                         total_difference for other in neighbors]
        return probabilities

    def select_neighbor(self, neighbors: List[Firefly], probabilities: List[float]) -> Firefly:
        '''
        Выбирает соседа на основе вероятностей методом рулетки.
        Параметры:
            neighbors (List[Firefly]): Список соседей.
            probabilities (List[float]): Список вероятностей.
        Возвращает:
            Firefly: Выбранный сосед.
        '''
        cumulative_probabilities = [
            sum(probabilities[:i + 1]) for i in range(len(probabilities))]
        rand = random.random()
        for i, prob in enumerate(cumulative_probabilities):
            if rand <= prob:
                return neighbors[i]

    def adjust_radius(self, firefly: Firefly, desired_neighbors: int):
        '''
        Корректирует радиус окрестности светлячка.
        Параметры:
            firefly (Firefly): Текущий светлячок.
            desired_neighbors (int): Целевое количество соседей.
        '''
        current_neighbors = len(self.calculate_neighbors(firefly))
        new_radius = firefly.radius + self.beta * \
            (desired_neighbors - current_neighbors)
        firefly.radius = min(self.max_radius, max(self.min_radius, new_radius))

    def optimize(self) -> Tuple[List[float], float]:
        '''
        Запускает процесс оптимизации.
        Возвращает:
            Tuple[List[float], float]: Лучшая позиция и значение целевой функции.
        '''
        self.initialize_fireflies()
        best_position = None
        best_value = float('inf')
        desired_neighbors = 5

        for iteration in range(self.max_iterations):
            for firefly in self.fireflies:
                function_value = self.fitness_function(*firefly.position)
                firefly.update_luciferin(function_value, self.rho, self.gamma)

            iteration_positions = []
            for firefly in self.fireflies:
                neighbors = self.calculate_neighbors(firefly)
                if neighbors:
                    probabilities = self.calculate_probabilities(
                        firefly, neighbors)
                    selected_neighbor = self.select_neighbor(
                        neighbors, probabilities)
                    firefly.move_towards(
                        selected_neighbor, self.delta, self.bounds)
                iteration_positions.append(firefly.position)
            self.positions_history.append(iteration_positions)

            for firefly in self.fireflies:
                self.adjust_radius(firefly, desired_neighbors)

            for firefly in self.fireflies:
                value = self.fitness_function(*firefly.position)
                if value < best_value:
                    best_value = value
                    best_position = firefly.position

            self.best_values.append(best_value)
            print(f"Итерация {iteration +
                  1}: Лучший результат = {best_value:.6f}")

        return best_position, best_value

    def plot_history(self):
        '''Отображает график сходимости.'''
        plt.plot(self.best_values)
        plt.title("Сходимость алгоритма роя светлячков")
        plt.xlabel("Итерации")
        plt.ylabel("Лучшее значение")
        plt.show()

    def visualize(self):
        '''Анимация перемещения светлячков с учётом яркости.'''
        fig, ax = plt.subplots()
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        def update(frame):
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Итерация {frame + 1}")
            positions = self.positions_history[frame]
            luciferin_values = [firefly.luciferin for firefly in self.fireflies]
            max_luciferin = max(luciferin_values)
            min_luciferin = min(luciferin_values)
            normalized_brightness = [
                (l - min_luciferin) / (max_luciferin - min_luciferin + 1e-9)
                for l in luciferin_values
            ]
            x_coords, y_coords = zip(*positions)
            ax.scatter(
                x_coords, y_coords,
                c="green",
                s=10,
                alpha=normalized_brightness
            )

        anim = FuncAnimation(
            fig, update, frames=len(self.positions_history), blit=False, interval=500, repeat=False
        )
        plt.show()


if __name__ == "__main__":
    swarm = FireflySwarm(
        fitness_function=goldstein_price,
        bounds=[(-2, 2), (-2, 2)],
        num_fireflies=100,
        max_iterations=200,
        beta=0.6,
        rho=0.4,
        delta=0.25,
        gamma=1.0,
        initial_radius=0.5
    )
    best_position, best_value = swarm.optimize()
    print(f"Оптимальное решение: {best_position}, Значение: {best_value:.6f}")
    swarm.plot_history()
    swarm.visualize()
