import time
import numpy as np
import matplotlib.pyplot as plt

ALLOWED_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tg": np.tan,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi
}

def constant(f, a, b, n=2000):
    xs = np.linspace(a, b, n)
    ys = np.array([f(x) for x in xs], dtype=float)

    if not np.all(np.isfinite(ys)):
        raise ValueError("Функция должна быть конечной на отрезке")
    slopes = np.abs(np.diff(ys) / np.diff(xs))
    return max(slopes.max() * 1.2, 1e-9)

def piyavskii(f, a, b, L, eps, max_iters=10000):
    xs = [a, b]
    ys = [float(f(a)), float(f(b))]
    start = time.perf_counter()

    for iteration in range(1, max_iters + 1):
        candidates = []
        for i in range(len(xs) - 1):
            xi, xj = xs[i], xs[i + 1]
            fi, fj = ys[i], ys[i + 1]
            if abs(fi - fj) > L * (xj - xi) + 1e-12:
                raise ValueError("Оценка L слишком маленькая")
            x_new = (xi + xj) / 2 + (fi - fj) / (2 * L)
            lower = (fi + fj - L * (xj - xi)) / 2
            candidates.append((lower, x_new, i))

        lower, x_new, i = min(candidates, key=lambda c: c[0])
        y_new = float(f(x_new))

        if not np.isfinite(y_new):
            raise ValueError("Получено некорректное значение функции")
        stop = y_new - lower < eps
        if not stop and not xs[i] < x_new < xs[i + 1]:
            raise ValueError("Не получается уточнить ломаную: проверь L")

        xs.insert(i + 1, x_new)
        ys.insert(i + 1, y_new)
        if stop:
            best = int(np.argmin(ys))
            elapsed = time.perf_counter() - start
            return xs[best], ys[best], iteration, elapsed, xs, ys
    raise RuntimeError("За указанное число итераций точность не достигнута")


formula = input("Введите f(x): ").strip().replace("^", "**")
a, b, eps = map(float, input("Введите a, b и eps через пробел: ").split())

if a >= b or eps <= 0:
    raise ValueError("Нужно a < b и eps > 0")

expression = compile(formula, "<function>", "eval")

def f(x):
    return eval(expression,{"__builtins__": {}},dict(ALLOWED_FUNCS, x=x))

L = constant(f, a, b)
x_min, f_min, iterations, elapsed, xs, ys = piyavskii(f, a, b, L, eps)

print(f"Оценка L: {L:.6f}")
print(f"x минимума: {x_min:.6f}")
print(f"f(x): {f_min:.6f}")
print(f"Итераций: {iterations}")
print(f"Всего проб в методе: {iterations + 2}")
print(f"Время поиска: {elapsed:.6f} с")

x_plot = np.linspace(a, b, 1000)
y_plot = [f(x) for x in x_plot]

broken_x = [xs[0]]
broken_y = [ys[0]]

for i in range(len(xs) - 1):
    t = (xs[i] + xs[i + 1]) / 2 + (ys[i] - ys[i + 1]) / (2 * L)
    h = (ys[i] + ys[i + 1] - L * (xs[i + 1] - xs[i])) / 2

    broken_x.extend([t, xs[i + 1]])
    broken_y.extend([h, ys[i + 1]])

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, label="Функция")
plt.plot(broken_x, broken_y, "--", color="green",
         label="Итоговая ломаная")
plt.scatter(x_min, f_min, color="red", s=60,
            label="Найденный минимум", zorder=3)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Метод ломаных")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
