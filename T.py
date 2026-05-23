# ===============================
# 1. NUMPY (np)
# ===============================
import numpy as np

# --- Атрибуты массива ---
a = np.array([[1,2,3],[4,5,6]])
a.ndim        # 2 - количество измерений
a.shape       # (2,3) - кортеж размеров
a.size        # 6 - общее число элементов
a.dtype       # dtype('int64') - тип элементов
a.itemsize    # 8 - размер одного элемента в байтах
a.nbytes      # 48 - общий объём памяти (size * itemsize)
a.T           # транспонированный массив (вид)

# --- Создание массивов ---
np.array([1,2,3])                # из списка
np.zeros((2,3))                  # массив нулей
np.ones((2,3))                   # массив единиц
np.full((2,3), 7)                # заполнить константой
np.eye(3)                        # единичная матрица 3x3
np.arange(0,10,2)                # [0,2,4,6,8]
np.linspace(0,1,5)               # [0, 0.25, 0.5, 0.75, 1]
np.random.rand(3,2)              # равномерное [0,1) форма (3,2)
np.random.randn(3,2)             # стандартное нормальное
np.random.randint(0,10, size=5)  # целые от 0 до 9

# --- Индексация и срезы ---
a[0,1]          # элемент строка 0, столбец 1
a[:, 1:3]       # срез: все строки, столбцы 1 и 2
a[a > 3]        # булева индексация (одномерный результат)
a[[0,1], [1,2]] # fancy indexing: (0,1) и (1,2) элементы

# --- Изменение формы ---
a.reshape(3,2)       # изменение формы (возвращает вид, где возможно)
a.flatten()          # копия в одномерный массив
a.ravel()            # вид (по возможности) в одномерный
a[:, np.newaxis]     # добавляет ось (становится (2,1,3) для a (2,3))

# --- Объединение и разбиение ---
np.concatenate((a,a), axis=0)   # вертикальное объединение
np.vstack((a,a))                # то же, но удобнее
np.hstack((a,a))                # горизонтальное объединение
np.split(a, 2, axis=0)          # разбить на 2 части по строкам

# --- Векторизованные операции (транслирование) ---
a + 5               # прибавить скаляр
a * a               # поэлементное умножение
a @ a.T             # матричное умножение (оператор @)
np.sin(a)           # поэлементная математическая функция

# --- Агрегация ---
np.sum(a)           # сумма всех элементов
np.mean(a, axis=0)  # среднее по столбцам
np.max(a, axis=1)   # максимум по строкам
np.cumsum(a)        # кумулятивная сумма (одномерный)
np.any(a > 3)       # хотя бы один True
np.all(a > 0)       # все True

# --- Линейная алгебра (np.linalg) ---
np.linalg.inv(a)        # обратная матрица (квадратная)
np.linalg.det(a)        # определитель
np.linalg.eig(a)        # собственные значения и векторы
np.linalg.solve(A, b)   # решение системы A x = b
np.linalg.lstsq(A, b)   # метод наименьших квадратов
np.linalg.norm(v)       # норма вектора/матрицы

# --- Работа с NaN ---
np.isnan(arr)            # маска на NaN
np.nanmean(arr)          # среднее, игнорируя NaN
np.nansum(arr)           # сумма, игнорируя NaN

# --- Особенности сравнения ---
np.nan == np.nan   # False
np.isnan(np.nan)   # True


# ===============================
# 2. PANDAS (pd)
# ===============================
import pandas as pd

# --- Создание Series / DataFrame ---
s = pd.Series([1,2,3], index=['a','b','c'])
df = pd.DataFrame({
    'col1': [1,2,3],
    'col2': [4,5,6]
}, index=['x','y','z'])

# --- Атрибуты ---
df.index          # индекс строк
df.columns        # индекс столбцов
df.values         # NumPy-представление данных
df.dtypes         # типы каждого столбца
df.shape          # (строки, столбцы)

# --- Доступ к элементам ---
df['col1']                # Series (один столбец)
df[['col1','col2']]       # DataFrame (несколько столбцов)
df.loc['x']               # строка по метке
df.iloc[0]                # строка по позиции
df.loc['x':'z', 'col1']   # срез по меткам (включая правую)
df.iloc[0:2, 0:1]         # срез по позициям
df.at['x', 'col1']        # быстрый доступ к скаляру по меткам
df.iat[0,0]               # быстрый доступ по позициям
df[df['col1'] > 1]        # булева фильтрация строк

# --- Добавление / удаление столбцов ---
df['new'] = [7,8,9]               # добавление
df.insert(1, 'new2', [0,0,0])     # вставка на позицию
df.drop('col1', axis=1)           # удалить столбец (axis=1)
df.drop('x', axis=0)              # удалить строку

# --- Работа с NaN ---
df.isna() / df.isnull()           # булева маска пропусков
df.dropna()                       # удалить строки с любым NaN
df.dropna(axis=1, how='all')      # удалить столбцы, где все NaN
df.fillna(0)                      # заменить NaN на 0
df.fillna(method='ffill')         # forward fill
df.interpolate()                  # интерполяция

# --- Агрегация ---
df.sum()                  # сумма по столбцам
df.mean(axis=1)           # среднее по строкам
df.describe()             # статистическая сводка
df.agg(['sum','mean'])    # несколько агрегаций
df['col1'].value_counts() # частоты уникальных значений

# --- Группировка ---
grouped = df.groupby('col1')
grouped['col2'].sum()                 # сумма по группам
grouped.agg({'col2': ['sum','mean']}) # разные агрегации
grouped.transform(lambda x: x/x.sum()) # сохраняет размерность
grouped.filter(lambda g: g['col2'].sum() > 10)

# --- Объединение ---
pd.concat([df1, df2], axis=0)            # склеить по строкам
pd.merge(left, right, on='key', how='inner') # SQL-подобное слияние
df1.join(df2, how='left')                # объединение по индексам

# --- Векторизованные операции и выравнивание ---
s1 + s2            # выравнивание по индексу, NaN где нет метки
df.add(s, axis=0)  # прибавить Series по строкам (axis=0)
df.add(df2, fill_value=0)  # перед сложением заменить NaN на 0

# --- Чтение/запись ---
pd.read_csv('file.csv')
df.to_csv('out.csv', index=False)
pd.read_excel('file.xlsx')
pd.read_sql('SELECT * FROM table', conn)


# ===============================
# 3. MATPLOTLIB (plt)
# ===============================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Настройки фигуры ---
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)      # 2 строки, 1 столбец, первый график

# --- Простой график (plot) ---
x = [1,2,3]; y = [4,5,6]
plt.plot(x, y, 'r--', label='line', linewidth=2, marker='o')
plt.xlabel('X'); plt.ylabel('Y'); plt.title('Title')
plt.legend(); plt.grid(True)

# --- Диаграмма рассеяния ---
plt.scatter(x, y, c='blue', s=50, alpha=0.7)

# --- Столбчатая диаграмма ---
plt.bar(['A','B','C'], [3,5,2], color='skyblue')
plt.barh(['A','B','C'], [3,5,2])   # горизонтальная

# --- Круговая диаграмма ---
plt.pie([25,40,35], labels=['A','B','C'], autopct='%1.1f%%')

# --- Гистограмма ---
data = np.random.randn(1000)
plt.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')

# --- Boxplot / Violinplot ---
plt.boxplot([data1, data2], labels=['X','Y'])
plt.violinplot([data1, data2], showmeans=True)

# --- Контурные и 2D-представления 3D-функций ---
X, Y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
Z = np.sin(X)*np.cos(Y)
plt.contour(X, Y, Z, levels=20, cmap='coolwarm')
plt.contourf(X, Y, Z, levels=20)      # залитые контуры
plt.pcolormesh(X, Y, Z, shading='auto')
plt.colorbar()

# --- 3D графики ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.plot_wireframe(X, Y, Z, color='black')
ax.scatter(xs, ys, zs, c=zs, cmap='hot')
ax.view_init(elev=30, azim=45)   # угол обзора

# --- Анимация (FuncAnimation) ---
from matplotlib.animation import FuncAnimation
def update(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
# ani.save('anim.mp4', writer='ffmpeg')   # сохранение

# --- Отображение / сохранение ---
plt.show()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.close()