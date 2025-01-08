from typing import Callable, List, Tuple
import random
import math


class SimulatedAnnealing:
    def __init__(self, func: Callable[..., float], bounds: List[Tuple[float, float]], k_max: int, T0: float):
        '''Инициализирует алгоритм имитации отжига с заданной целевой функцией, 
        границами поиска, максимальным числом итераций и начальной температурой.
        Параметры:
            func (Callable[[float, float], float]): Целевая функция, минимизация которой требуется.
            bounds (List[Tuple[float, float]]): Границы для каждой переменной в формате [(min, max), ...].
            k_max (int): Максимальное количество итераций.
            T0 (float): Начальная температура.
        '''
        self.func = func
        self.bounds = bounds
        self.k_max = k_max
        self.T0 = T0
        self.D = len(bounds)  # Размерность пространства состояний
        self.current_solution = self.random_solution()
        self.current_cost = self.func(*self.current_solution)

    def random_solution(self) -> List[float]:
        '''Генерирует случайное начальное решение в пределах указанных границ.
        Возвращает:
            List[float]: Список значений переменных, представляющих решение.
        '''
        return [random.uniform(b[0], b[1]) for b in self.bounds]

    @staticmethod
    def cauchy_distribution(x: float, main_x: float, temperature: float) -> float:
        '''Вычисляет распределение Коши для данной точки.
        Параметры:
            x (float): Точка, в которой вычисляется распределение.
            main_x (float): Основная точка, определяющая центр распределения.
            temperature (float): Текущая температура.
        Возвращает:
            float: Значение распределения.
        '''
        return (1 / math.pi) * temperature / ((x - main_x) ** 2 + temperature ** 2)

    def generate_solution(self, temperature: float) -> List[float]:
        '''Генерирует новое решение на основе текущего, используя распределение Коши.
        Параметры:
            temperature (float): Текущая температура.
        Возвращает:
            List[float]: Новое решение.
        '''
        new_solution = []
        for i in range(self.D):
            while True:
                main_x = self.current_solution[i]
                new_x = random.uniform(self.bounds[i][0], self.bounds[i][1])
                p_distribute = self.cauchy_distribution(
                    new_x, main_x, temperature)
                p = random.random()
                if p <= p_distribute:
                    new_solution.append(new_x)
                    break
        return new_solution

    def temperature(self, k: int) -> float:
        '''Вычисляет температуру на текущей итерации.
        Параметры:
            k (int): Текущий номер итерации.
        Возвращает:
            float: Значение температуры.
        '''
        return self.T0 / (k ** (1 / self.D))

    def acceptance_probability(self, e_old: float, e_new: float, T: float) -> float:
        '''Вычисляет вероятность принятия нового решения.
        Параметры:
            e_old (float): Энергия текущего решения.
            e_new (float): Энергия нового решения.
            T (float): Текущая температура.
        Возвращает:
            float: Вероятность принятия нового решения.
        '''
        if e_new < e_old:
            return 1.0
        return math.exp(-(e_new - e_old) / T)

    def optimize(self) -> Tuple[List[float], float]:
        '''Запускает алгоритм оптимизации для поиска минимального значения функции.
        Возвращает:
            Tuple[List[float], float]: Координаты минимального решения и значение функции в этой точке.
        '''
        best_solution = self.current_solution
        best_cost = self.current_cost
        k = 1
        while k <= self.k_max:
            T = self.temperature(k)
            new_solution = self.generate_solution(T)
            new_cost = self.func(*new_solution)
            print(f"Итерация: {k}/{self.k_max}")
            print(f"Температура: {T:.12f}")
            print("Текущее решение:", self.current_solution)
            print(f"Текущая стоимость: {self.current_cost:.12f}")
            print("Новое решение:", new_solution)
            print(f"Новое значение функции: {new_cost:.12f}")
            acceptance_probability = self.acceptance_probability(
                self.current_cost, new_cost, T)
            print(f"Вероятность принятия нового решения: {
                  acceptance_probability:.12f}")
            random_num = random.random()
            print(f"Сгенерированное число: {random_num:.12f}")
            if acceptance_probability >= random_num:
                if new_cost - self.current_cost > 0:
                    print('\033[95m' + 'Принято худшее решение' + '\033[0m')
                self.current_solution = new_solution
                self.current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
            print('-' * 40)
            k += 1
        return best_solution, best_cost


# Функция Гольдштейна-Прайса
def goldstein_price(x: float, y: float) -> float:
    term1 = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    term2 = (30 + (2*x - 3*y)**2 * (18 - 32*x +
             12*x**2 + 48*y - 36*x*y + 27*y**2))
    return term1 * term2


bounds = [(-2, 2), (-2, 2)]
solution = SimulatedAnnealing(goldstein_price, bounds, k_max=2500, T0=20)
result = solution.optimize()
print("Координаты минимума:", result[0])
print("Минимальное значение функции:", result[1])
print("Координаты текущего решения:", solution.current_solution)
print("Стоимость текущего решения:", solution.current_cost)
