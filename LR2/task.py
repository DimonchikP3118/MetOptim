import time
import numpy as np
import matplotlib.pyplot as plt

ALLOWED_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tg": np.tan,
    "tan": np.tan,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e
}


def constant(f, a, b, n=2000):
    xs = np.linspace(a, b, n)
    ys = np.array([f(x) for x in xs], dtype=float)

    if not np.all(np.isfinite(ys)):
        raise ValueError("Функция должна быть конечной на отрезке")

    slopes = np.abs(np.diff(ys) / np.diff(xs))
    return max(slopes.max() * 1.2, 1e-9)


def piyavskii(f, a, b, L, eps, max_iters=10000):
    if not all(np.isfinite([a, b, L, eps])):
        raise ValueError("Границы, L и точность должны быть конечными")

    if a >= b or L <= 0 or eps <= 0 or max_iters < 1:
        raise ValueError("Нужно a < b, L > 0, eps > 0, max_iters >= 1")

    xs = [a, b]
    ys = [float(f(a)), float(f(b))]

    if not np.all(np.isfinite(ys)):
        raise ValueError("Функция должна быть конечной на концах отрезка")

    start = time.perf_counter()
    reached = False

    for iteration in range(1, max_iters + 1):
        candidates = []

        for i in range(len(xs) - 1):
            xi, xj = xs[i], xs[i + 1]
            fi, fj = ys[i], ys[i + 1]

            if abs(fi - fj) > L * (xj - xi) + 1e-12:
                raise ValueError("L слишком маленькая: нарушено условие Липшица")

            x_new = (xi + xj) / 2 + (fi - fj) / (2 * L)
            lower = (fi + fj - L * (xj - xi)) / 2
            candidates.append((lower, x_new, i))

        lower, x_new, i = min(candidates, key=lambda c: c[0])
        y_new = float(f(x_new))

        if not np.isfinite(y_new):
            raise ValueError("Получено некорректное значение функции")

        gap = y_new - lower
        tolerance = 1e-12 * max(1.0, abs(y_new), abs(lower))

        if gap < -tolerance:
            raise ValueError(
                f"L слишком маленькая: функция ниже ломаной "
                f"(зазор {gap:.6g}). Увеличьте L и повторите поиск"
            )

        gap = max(gap, 0.0)
        reached = gap < eps

        if not reached and not xs[i] < x_new < xs[i + 1]:
            raise ValueError("Не получается уточнить ломаную: проверьте L")

        xs.insert(i + 1, x_new)
        ys.insert(i + 1, y_new)

        if reached:
            break

    best = int(np.argmin(ys))
    elapsed = time.perf_counter() - start

    return xs[best], ys[best], iteration, elapsed, xs, ys, reached


def main():
    formula = input("Введите f(x): ").strip().replace("^", "**")

    try:
        expression = compile(formula, "<function>", "eval")
    except SyntaxError:
        print("Ошибка: неверная запись формулы. Проверьте скобки и знаки операций")
        return

    unknown = set(expression.co_names) - set(ALLOWED_FUNCS) - {"x"}

    if unknown:
        print("Ошибка: неизвестные имена: " + ", ".join(sorted(unknown)))
        print("Доступны: x, " + ", ".join(ALLOWED_FUNCS))
        return

    try:
        a, b, eps = map(float, input("Введите a, b и eps через пробел: ").split())

        if not all(np.isfinite([a, b, eps])) or a >= b or eps <= 0:
            raise ValueError("Нужно конечные a < b и eps > 0")

        def f(x):
            return eval(expression, {"__builtins__": {}}, dict(ALLOWED_FUNCS, x=x))

        L = constant(f, a, b)

        x_min, f_min, iterations, elapsed, xs, ys, reached = piyavskii(
            f, a, b, L, eps
        )

        if reached:
            print("Точность достигнута")
        else:
            print("Лимит итераций исчерпан. Точность не достигнута")

        print(f"Оценка L: {L:.6f}")
        print(f"Лучшая найденная точка: {x_min:.6f}")
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
        plt.plot(broken_x, broken_y, "--", color="green", label="Итоговая ломаная")
        plt.scatter(
            x_min, f_min, color="red", s=60,
            label="Лучшая найденная точка", zorder=3
        )

        plt.xlabel("x")
        plt.ylabel("f(x)")

        if reached:
            plt.title("Метод ломаных")
        else:
            plt.title("Лимит итераций: точность не достигнута")

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as error:
        print(f"Ошибка ({type(error).__name__}): {error}")


if __name__ == "__main__":
    main()
