QUERY = {'Фильм': ('Актёры', 'Название'),
         'Многосерийный фильм': ('Актёры', 'Название', 'Количество сезонов', 'Количество эпизодов'),
         'Полнометражный фильм': ('Актёры', 'Название', 'Длительность'),
         'Актёр': ('ФИО', ),
         'Режиссёр': ('ФИО', 'Руководит')}

FUNCTION = {'Актёры': ('contains', 'does not contain'),
            'Название': ('contains', 'does not contain', 'is',
                         'is not', 'begins with', 'ends with'),
            'Количество сезонов': ('is', 'is greater then', 'is less then'),
            'Количество эпизодов': ('is', 'is greater then', 'is less then'),
            'Длительность': ('is', 'is greater then', 'is less then'),
            'ФИО': ('contains', 'does not contain', 'is',
                    'is not', 'begins with', 'ends with'),
            'Руководит': ('contains', 'does not contain')}


class MovieIndustry:
    def __init__(self, *args, **kwargs):
        raise TypeError(f"Can't instantiate abstract class {
                        __class__.__name__}")


class Film(MovieIndustry):
    def __init__(self, name, actors):
        self._name = name
        self._actors = list(actors)

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"{__class__.__name__}('{self._name}', {self._actors})"

    def __eq__(self, other):
        if type(other) is __class__:
            return other._name == self._name
        elif type(other) is str:
            return other == self._name
        return NotImplemented
    
    def __contains__(self, obj):
        return obj in self._name
    
    def has_actor(self, actor):
        return actor in self._actors


class SerialFilm(Film):
    def __init__(self, name, actors, num_seasons, num_episodes):
        super().__init__(name, actors)
        self._num_seasons = num_seasons
        self._num_episodes = num_episodes

    def __repr__(self):
        return f"{__class__.__name__}('{self._name}', {self._actors}, {self._num_seasons}, {self._num_episodes})"


class FeatureFilm(Film):
    def __init__(self, name, actors, length):
        super().__init__(name, actors)
        self._length = length

    def __repr__(self):
        return f"{__class__.__name__}('{self._name}', {self._actors}, {self._length})"


class Actor(MovieIndustry):
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"{__class__.__name__}('{self._name}')"

    def __eq__(self, other):
        if type(other) is __class__:
            return other._name == self._name
        elif type(other) is str:
            return other == self._name
        return NotImplemented

    def __contains__(self, obj):
        return obj in self._name

    def startswith(self, value):
        return self._name.startswith(value)

    def endswith(self, value):
        return self._name.endswith(value)


class FilmDirector(MovieIndustry):
    def __init__(self, name, films):
        self._name = name
        self._films = list(films)

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"{__class__.__name__}('{self._name}', {self._films})"

    def __eq__(self, other):
        if type(other) is __class__:
            return other._name == self._name
        elif type(other) is str:
            return other == self._name
        return NotImplemented

    def __contains__(self, obj):
        return obj in self._name

    def startswith(self, value):
        return self._name.startswith(value)

    def endswith(self, value):
        return self._name.endswith(value)

    def has_movie(self, value):
        return value in self._films


class Queries:
    def __init__(self, cls, slot, function, value):
        self._cls = cls
        self._slot = slot
        self._value = value
        match function:
            case 'contains':
                self._func = __class__.contains
            case 'does not contain':
                self._func = __class__.does_not_contains
            case 'is':
                self._func = __class__.equal
            case 'is not':
                self._func = __class__.not_equal
            case 'begins with':
                self._func = __class__.startswith
            case 'ends with':
                self._func = __class__.endswith
            case 'is greater then':
                self._func = function
            case 'is less then':
                self._func = function
        self._result = None

    def find(self):
        match self._cls:
            case 'Фильм':
                if self._slot == 'Название':
                    found_films = list(filter(lambda object: self._func(self, object), feature_films + serial_films))
                elif self._func == __class__.contains:
                    found_films = list(filter(lambda object: object.has_actor(self._value), feature_films + serial_films))
                else:
                    found_films = list(filter(lambda object: not object.has_actor(self._value), feature_films + serial_films))
                for film in found_films:
                    print(f'• {film} ({type(film).__name__})')
                    for film_director in film_directors:
                        if film_director.has_movie(film):
                            print(f'    • {film_director} ({type(film_director).__name__})')
            case 'Многосерийный фильм':
                if self._slot == 'Актёры' and self._func == __class__.contains:
                    found_films = list(filter(lambda object: object.has_actor(self._value), serial_films))
                elif self._slot == 'Актёры' and self._func == __class__.does_not_contains:
                    found_films = list(filter(lambda object: not object.has_actor(self._value), serial_films))
                elif self._slot == 'Количество сезонов' and self._func == __class__.equal:
                    found_films = list(filter(lambda object: object._num_seasons == int(self._value), serial_films))
                elif self._slot == 'Количество сезонов' and self._func == 'is greater then':
                    found_films = list(filter(lambda object: object._num_seasons > int(self._value), serial_films))
                elif self._slot == 'Количество сезонов' and self._func == 'is less then':
                    found_films = list(filter(lambda object: object._num_seasons < int(self._value), serial_films))
                elif self._slot == 'Количество эпизодов' and self._func == __class__.equal:
                    found_films = list(filter(lambda object: object._num_episodes == int(self._value), serial_films))
                elif self._slot == 'Количество эпизодов' and self._func == 'is greater then':
                    found_films = list(filter(lambda object: object._num_episodes > int(self._value), serial_films))
                elif self._slot == 'Количество эпизодов' and self._func == 'is less then':
                    found_films = list(filter(lambda object: object._num_episodes < int(self._value), serial_films))
                else:
                    found_films = list(filter(lambda object: self._func(self, object), serial_films))
                for film in found_films:
                    print(f'• {film} ({type(film).__name__})')
                    for film_director in film_directors:
                        if film_director.has_movie(film):
                            print(f'    • {film_director} ({type(film_director).__name__})')
            case 'Полнометражный фильм':
                if self._slot == 'Актёры' and self._func == __class__.contains:
                    found_films = list(filter(lambda object: object.has_actor(self._value), feature_films))
                elif self._slot == 'Актёры' and self._func == __class__.does_not_contains:
                    found_films = list(filter(lambda object: not object.has_actor(self._value), feature_films))
                elif self._slot == 'Длительность' and self._func == __class__.equal:
                    found_films = list(filter(lambda object: object._length == int(self._value), feature_films))
                elif self._slot == 'Длительность' and self._func == 'is greater then':
                    found_films = list(filter(lambda object: object._length > int(self._value), feature_films))
                elif self._slot == 'Длительность' and self._func == 'is less then':
                    found_films = list(filter(lambda object: object._length < int(self._value), feature_films))
                else:
                    found_films = list(filter(lambda object: self._func(self, object), feature_films))
                for film in found_films:
                    print(f'• {film} ({type(film).__name__})')
                    for film_director in film_directors:
                        if film_director.has_movie(film):
                            print(f'    • {film_director} ({type(film_director).__name__})')
            case 'Актёр':
                found_actors = list(
                    filter(lambda object: self._func(self, object), actors))
                for actor in found_actors:
                    print(f'• {actor} ({type(actor).__name__})')
                    for film in serial_films + feature_films:
                        if film.has_actor(actor):
                            print(f'    • {film} ({type(film).__name__})')
                            for film_director in film_directors:
                                if film_director.has_movie(film):
                                    print(f'\t• {film_director} ({type(film_director).__name__})')
            case 'Режиссёр':
                if self._slot == 'ФИО':
                    self._result = list(
                        filter(lambda object: self._func(self, object), film_directors))
                elif self._func == __class__.contains:
                    self._result = list(
                        filter(lambda object: object.has_movie(self._value), film_directors))
                else:
                    self._result = list(
                        filter(lambda object: not object.has_movie(self._value), film_directors))
                for item in self._result:
                    print(f'• {str(item)} ({type(item).__name__})')

    def contains(self, object):
        return self._value in object

    def does_not_contains(self, object):
        return self._value not in object

    def equal(self, object):
        return self._value == object

    def not_equal(self, object):
        return self._value != object

    def startswith(self, object):
        return object.startswith(self._value)

    def endswith(self, object):
        return object.endswith(self._value)


def make_query():
    '''Функция для написания запроса'''
    print("\033[4mВыберите класс:\033[0m")
    for num, cls in enumerate(QUERY, 1):
        print(num, '-', '\033[93m' + cls + '\033[0m')
    cls_num = input()
    if cls_num not in map(str, range(1, len(QUERY) + 1)):
        raise TypeError('Некорректный номер класса')
    cls = list(QUERY.keys())[int(cls_num) - 1]

    print("\033[4mВыберите слот:\033[0m")
    for num, slot in enumerate(QUERY[cls], 1):
        print(num, '-', '\033[94m' + slot + '\033[0m')
    slot_num = input()
    if slot_num not in map(str, range(1, len(QUERY[cls]) + 1)):
        raise TypeError('Некорректный номер слота')
    slot = QUERY[cls][int(slot_num) - 1]

    print("\033[4mВыберите функцию запроса:\033[0m")
    for num, func in enumerate(FUNCTION[slot], 1):
        print(num, '-', '\033[93m' + func + '\033[0m')
    func_num = input()
    if func_num not in map(str, range(1, len(FUNCTION[slot]) + 1)):
        raise TypeError('Некорректный номер функции')
    func = FUNCTION[slot][int(func_num) - 1]

    value = input("\033[4mВведите значение запроса:\033[0m ")

    query_obj = Queries(cls, slot, func, value)
    query_obj.find()


actors = [Actor('Bryan Cranston'), Actor('Anna Gunn'),
          Actor('Aaron Paul'), Actor('Dean Norris'),
          Actor('Betsy Brandt'), Actor('RJ Mitte'),
          Actor('Bob Odenkirk'), Actor('Giancarlo Esposito'),
          Actor('Jonathan Banks'), Actor('Steven Michael Quezada'),
          Actor('Cillian Murphy'), Actor('Emily Blunt'),
          Actor('Matt Damon'), Actor('Robert Downey Jr.'),
          Actor('Florence Pugh'), Actor('Josh Hartnett'),
          Actor('David Krumholtz'), Actor('Benny Safdie'),
          Actor('Alden Ehrenreich'), Actor('Kenneth Branagh'),
          Actor('Leonardo DiCaprio'), Actor('Jonah Hill'),
          Actor('Margot Robbie'), Actor('Kyle Chandler'),
          Actor('Rob Reiner'), Actor('P.J. Byrne'),
          Actor('Jon Bernthal'), Actor('Cristin Milioti'),
          Actor('Jean Dujardin'), Actor('Matthew McConaughey'),
          Actor('Ryan Gosling'), Actor('America Ferrera'),
          Actor('Ariana Greenblatt'), Actor('Kate McKinnon'),
          Actor('Issa Rae'), Actor('Will Ferrell'),
          Actor('Michael Cera'), Actor('Simu Liu'),
          Actor('Alexandra Shipp'), Actor('Joseph Gordon-Levitt'),
          Actor('Elliot Page'), Actor('Tom Hardy'),
          Actor('Ken Watanabe'), Actor('Dileep Rao'),
          Actor('Tom Berenger'), Actor('Marion Cotillard'),
          Actor('Pete Postlethwaite'), Actor('Paul Anderson'),
          Actor('Sophie Rundle'), Actor('Helen McCrory'),
          Actor('Ned Dennehy'), Actor('Finn Cole'),
          Actor("Natasha O'Keeffe"), Actor('Ian Peck'),
          Actor('Harry Kirton'), Actor('Packy Lee'),
          Actor('Matthew Fox'), Actor('Evangeline Lilly'),
          Actor('Josh Holloway'), Actor("Terry O'Quinn"),
          Actor('Naveen Andrews'),Actor('Jorge Garcia'),
          Actor('Michael Emerson'),Actor('Emilie de Ravin'),
          Actor('Kim Yoon-jin'),Actor('Daniel Dae Kim'),
          Actor('Henry Ian Cusick'),Actor('Dominic Monaghan')]

serial_films = [SerialFilm('Breaking Bad', actors[:10], 5, 62),
                SerialFilm('Peaky Blinders', [actors[10]] + actors[47:56], 6, 36),
                SerialFilm('Lost', actors[56:68], 6, 121)]

feature_films = [FeatureFilm('Oppenheimer', actors[10:20], 180),
                 FeatureFilm('The Wolf of Wall Street', actors[20:30], 172),
                 FeatureFilm('Barbie', [actors[22]] + actors[30:39], 104),
                 FeatureFilm('Inception', [actors[20]] + [actors[10]] + actors[39:47], 148)]

film_directors = [FilmDirector('Michelle MacLaren', [serial_films[0]]),
                  FilmDirector('Adam Bernstein', [serial_films[0]]),
                  FilmDirector('Vince Gilligan', [serial_films[0]]),
                  FilmDirector('Christopher Nolan', [
                               feature_films[0], feature_films[3]]),
                  FilmDirector('Martin Scorsese', [feature_films[1]]),
                  FilmDirector('Greta Gerwig', [feature_films[2]]),
                  FilmDirector('Anthony Byrne', [serial_films[1]]),
                  FilmDirector('Colm McCarthy', [serial_films[1]]),
                  FilmDirector('Tim Mielants', [serial_films[1]]),
                  FilmDirector('David Caffrey', [serial_films[1]]),
                  FilmDirector('Otto Bathurst', [serial_films[1]]),
                  FilmDirector('Tom Harper', [serial_films[1]]),
                  FilmDirector('Jack Bender', [serial_films[2]]),
                  ]

make_query()

# print('Список актеров: ', len(actors))
# print('Множество актёров: ', len(set(actor._name for actor in actors)))
# print('Список режиссёров: ', len(film_directors))
# print('Множество режиссёров: ', len(set(film_director._name for film_director in film_directors)))
# print(repr(serial_films[2]))