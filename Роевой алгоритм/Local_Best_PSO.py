import random
from typing import List, Tuple, Callable


def goldstein_price(x: float, y: float) -> float:
    '''Функция Голдштейна-Прайса для оптимизации.'''
    term1 = (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2))
    term2 = (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))
    return term1 * term2


class Particle:
    def __init__(self, bounds: List[Tuple[float, float]], fitness_function: Callable[..., float]):
        '''
        Инициализирует частицу с случайными позицией и скоростью.
        Параметры:
            bounds (List[Tuple[float, float]]): Ограничения для координат каждой частицы.
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
        '''
        self.position = [random.uniform(*bound) for bound in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = self.position[:]
        self.fitness_function = fitness_function
        self.best_value = self.fitness_function(*self.position)

    def __str__(self) -> str:
        '''Возвращает строковое представление текущего состояния частицы.'''
        return f"(Координаты: {[f'{pos:.4f}' for pos in self.position]}; " + \
               f"Скорость: {[f'{vel:.4f}' for vel in self.velocity]}; "  + \
               f"Лучшие координаты: {[f'{b_pos:.4f}' for b_pos in self.best_position]}; " + \
               f"Лучшее значение функции: {self.best_value:.4f})."

    def update_velocity(self, local_best_position: List[float], c1: float = 2.0, c2: float = 2.0) -> None:
        '''
        Обновляет скорость частицы на основе её лучшей позиции и локальной лучшей позиции.
        Параметры:
            local_best_position (List[float]): Локальная лучшая позиция.
            c1 (float): Коэффициент когнитивного компонента.
            c2 (float): Коэффициент социального компонента.
        '''
        for i in range(len(self.position)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (local_best_position[i] - self.position[i])
            self.velocity[i] += cognitive + social

    def update_position(self, bounds: List[Tuple[float, float]]) -> None:
        '''
        Обновляет позицию частицы с учётом ограничений и обновляет её лучшую позицию.
        Параметры:
            bounds (List[Tuple[float, float]]): Ограничения для координат.
        '''
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            self.position[i] = max(min(self.position[i], bounds[i][1]), bounds[i][0])
        current_value = self.fitness_function(*self.position)
        if current_value < self.best_value:
            self.best_position = self.position[:]
            self.best_value = current_value


class Swarm:
    def __init__(self, fitness_function: Callable[..., float], bounds: List[Tuple[float, float]],
                 num_particles: int, max_iterations: int):
        '''
        Инициализирует рой частиц для оптимизации.
        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            bounds (List[Tuple[float, float]]): Ограничения для координат.
            num_particles (int): Количество частиц в рое.
            max_iterations (int): Максимальное количество итераций для оптимизации.
        '''
        self.particles = [Particle(bounds, fitness_function) for _ in range(num_particles)]
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations

    def get_local_best(self, index: int) -> List[float]:
        '''
        Находит лучшую позицию среди соседей данной частицы.
        Параметры:
            index (int): Индекс текущей частицы.
        Возвращает:
            List[float]: Лучшая позиция среди соседей.
        '''
        neighbors_indices = [(index - 1) % len(self.particles), index, (index + 1) % len(self.particles)]
        neighbors = [self.particles[i] for i in neighbors_indices]
        best_neighbor = min(neighbors, key=lambda p: p.best_value)
        return best_neighbor.best_position

    def optimize(self) -> Tuple[List[float], float]:
        '''
        Выполняет оптимизацию, обновляя позиции и скорости частиц.
        Возвращает:
            Tuple[List[float], float]: Лучшая позиция и значение целевой функции.
        '''
        for _ in range(self.max_iterations):
            for i, particle in enumerate(self.particles):
                local_best_position = self.get_local_best(i)
                particle.update_velocity(local_best_position)
            for particle in self.particles:
                particle.update_position(bounds)
        best_particle = min(self.particles, key=lambda p: p.best_value)
        return best_particle.best_position, best_particle.best_value


bounds = [(-2, 2), (-2, 2)]
num_particles = 500  # Количество частиц
max_iterations = 500  # Максимальное количество итераций
swarm = Swarm(goldstein_price, bounds, num_particles, max_iterations)
best_position, best_value = swarm.optimize()
print("Лучшая позиция:", best_position)
print(f"Лучшее значение функции: {best_value:.12f}")
