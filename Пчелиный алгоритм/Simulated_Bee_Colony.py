import random
import math
import itertools
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Point:
    def __init__(self, *args):
        self.coordinates = args

    def euclidean_distance(self, other_point):
        '''Вычисляет евклидово расстояние между текущей точкой и другой точкой.'''
        if not isinstance(other_point, __class__):
            raise ValueError('Евклидово расстояние может быть рассчитано только между экземплярами Point.')
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(self, other_point)))

    def __repr__(self):
        return f"Point{self.coordinates}"

    def __str__(self):
        return str(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]


class BeeColony:
    def __init__(self, fitness_function, bounds: List[Tuple[float, float]], scouts: int, k: int,
                 distance_threshold: float, l: float, max_iterations: int, maximize: bool = False):
        '''
        Инициализирует алгоритм пчелиной колонии для оптимизации.
        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            bounds (List[Tuple[float, float]]): Ограничения для каждой координаты в виде [(min1, max1), ..., (minD, maxD)].
            scouts (int): Количество пчел-разведчиков.
            k (int): Количество итераций без улучшения для остановки алгоритма.
            distance_threshold (float): Максимальное евклидово расстояние для объединения точек.
            l (float): Размер области локального поиска.
            max_iterations (int): Максимальное количество итераций алгоритма.
            maximize (bool): Определяет, ищется максимум (True) или минимум функции (False).
        '''
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.scouts = scouts
        self.k = k
        self.distance_threshold = distance_threshold
        self.l = l
        self.max_iterations = max_iterations
        self.maximize = maximize
        self.history = list()

    def compare(self, a, b):
        '''Сравнение значений функции с учетом типа оптимизации.'''
        return a > b if self.maximize else a < b

    def optimize(self) -> Tuple[Point, float]:
        '''
        Выполняет оптимизацию с использованием алгоритма пчелиной колонии.
        Возвращает:
            Tuple[Point, float]: Лучшая найденная точка и значение целевой функции в ней.
        '''
        scouts = [
            Point(*[random.uniform(bounds[0], bounds[1]) for bounds in self.bounds])
            for _ in range(self.scouts)
        ]
        best_global_value = float("-inf") if self.maximize else float("inf")
        best_global_point = None
        num_iteration = 0
        stagnation_count = 0

        while num_iteration < self.max_iterations:
            print(f"Итерация №{num_iteration}")
            num_iteration += 1
            iteration_data = []

            values = [self.fitness_function(*point) for point in scouts]
            best_local_index = max(range(len(values)), key=lambda i: values[i]) \
                if self.maximize else min(range(len(values)), key=lambda i: values[i])
            best_local_point = scouts[best_local_index]
            best_local_value = values[best_local_index]
            if self.compare(best_local_value, best_global_value):
                best_global_value = best_local_value
                best_global_point = best_local_point
                stagnation_count = 0
            else:
                stagnation_count += 1

            combined_regions = []
            used_indices = set()
            for i, point_i in enumerate(scouts):
                if i in used_indices:
                    continue
                region = [point_i]
                for j, point_j in enumerate(scouts):
                    if j != i and j not in used_indices and \
                            point_i.euclidean_distance(point_j) <= self.distance_threshold:
                        region.append(point_j)
                        used_indices.add(j)
                used_indices.add(i)
                combined_regions.append(region)
            print(f"Количество подобластей: {len(combined_regions)}")

            new_scouts = []
            for region in combined_regions:
                center = max(region, key=lambda p: self.fitness_function(*p)) \
                    if self.maximize else min(region, key=lambda p: self.fitness_function(*p))
                search_area = [
                    (max(self.bounds[i][0], center[i] - self.l), min(self.bounds[i][1], center[i] + self.l))
                    for i in range(len(self.bounds))
                ]
                local_scouts = [center] + [
                    Point(*[random.uniform(area[0], area[1]) for area in search_area])
                    for _ in range(self.scouts - 1)
                ]
                iteration_data.append(local_scouts)
                local_values = [self.fitness_function(*point) for point in local_scouts]
                best_local_index = max(range(len(local_values)), key=lambda i: local_values[i]) \
                    if self.maximize else min(range(len(local_values)), key=lambda i: local_values[i])
                new_scouts.append(local_scouts[best_local_index])

            self.history.append(iteration_data)
            scouts = new_scouts
            if stagnation_count >= self.k:
                break
            print(f"Лучшая позиция: {best_global_point}")
            print(f"Лучшее значение функции: {best_global_value:.6f}")
        return best_global_point, best_global_value

    def visualize(self):
        '''Визуализирует работу пчел.'''
        fig, ax = plt.subplots()
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        def update(frame):
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Итерация {frame + 1}")
            iteration_data = self.history[frame]
            for region_points in iteration_data:
                x_coords = [p[0] for p in region_points]
                y_coords = [p[1] for p in region_points]
                color = next(colors)
                ax.scatter(x_coords, y_coords, c=color, s=10, label="Region Points")
            ax.legend(loc='upper right')
        anim = FuncAnimation(fig, update, frames=len(self.history), blit=False, interval=500, repeat=False)
        plt.show()


def goldstein_price(x: float, y: float) -> float:
    '''Функция Голдштейна-Прайса для оптимизации.'''
    term1 = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
    term2 = (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return term1 * term2


def sphere(x, y):
    '''Функция сферы для оптимизации.'''
    return x ** 2 + y ** 2


colony = BeeColony(
    fitness_function=goldstein_price,
    bounds=[(-20, 20), (-20, 20)],
    scouts=100,
    k=100,
    distance_threshold=1.5,
    l=0.5,
    max_iterations=1000,
    maximize=False
)

best_position, best_value = colony.optimize()
colony.visualize()
print("Лучшая позиция:", best_position)
print(f"Лучшее значение функции: {best_value:.6f}")
