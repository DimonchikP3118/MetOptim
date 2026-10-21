import numpy as np

np.set_printoptions(precision=4, suppress=True)
EPS = 1e-9


def read_data(file_name):
    with open(file_name, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2 or lines[0] not in ('max', 'min'):
        raise ValueError("Первая строка должна быть max или min")

    is_max = lines[0] == 'max'
    func_coeffs = list(map(float, lines[1].split()))
    n = len(func_coeffs)

    table = []
    free_vars = set()

    for line in lines[2:]:
        raw = line.split()

        if raw[0] == 'free':
            for value in raw[1:]:
                j = int(value) - 1
                if not 0 <= j < n:
                    raise ValueError("Неверный номер свободной переменной")
                free_vars.add(j)
            continue

        if len(raw) != n + 2 or raw[-2] not in ('=', '>=', '<='):
            raise ValueError("Неверная запись ограничения")

        coeffs = list(map(float, raw[:-2]))
        sign = raw[-2]
        b = float(raw[-1])
        table.append((coeffs, sign, b))

    return is_max, func_coeffs, table, free_vars


def to_canonical(table, n_vars):
    extra_count = sum(sign != '=' for _, sign, _ in table)
    rows = []
    b_col = []
    extra = 0

    for coeffs, sign, b in table:
        if len(coeffs) != n_vars or sign not in ('=', '<=', '>='):
            raise ValueError("Неверная запись ограничения")

        row = list(coeffs) + [0.0] * extra_count

        if sign != '=':
            row[n_vars + extra] = 1.0 if sign == '<=' else -1.0
            extra += 1

        # Правая часть равенства должна быть неотрицательной
        if b < 0:
            row = [-v for v in row]
            b = -b

        rows.append(row)
        b_col.append(b)

    return rows, b_col, n_vars + extra_count


def find_basis(A):
    m, n = A.shape
    basis = []
    used = set()

    for i in range(m):
        found = None

        for j in range(n):
            if j in used:
                continue

            if abs(A[i, j] - 1.0) < EPS and all(
                abs(A[k, j]) < EPS for k in range(m) if k != i
            ):
                found = j
                break

        if found is None:
            return None

        basis.append(found)
        used.add(found)

    return basis


def prepare_basis(A, b, basis):
    T = np.array(A, dtype=float).copy()
    rhs = np.array(b, dtype=float).copy()
    m = len(rhs)

    if len(basis) != m:
        raise ValueError("Неполный базис")

    for i, j in enumerate(basis):
        k = i + np.argmax(np.abs(T[i:, j]))

        if abs(T[k, j]) <= EPS:
            raise ValueError("Вырожденная базисная матрица")

        T[[i, k]] = T[[k, i]]
        rhs[[i, k]] = rhs[[k, i]]

        pivot = T[i, j]
        T[i] /= pivot
        rhs[i] /= pivot

        for k in range(m):
            if k != i:
                factor = T[k, j]
                T[k] -= factor * T[i]
                rhs[k] -= factor * rhs[i]

    return T, rhs


def simplex(c, A, b, basis, max_iter=10000):
    c = np.array(c, dtype=float)
    basis = list(basis)
    n = len(c)

    # Индексы небазисных переменных
    free = [j for j in range(n) if j not in basis]

    T, rhs = prepare_basis(A, b, basis)

    M = T[:, free].copy()
    p = c[free] - c[basis] @ M
    Q = float(c[basis] @ rhs)

    R = np.eye(len(rhs))

    for _ in range(max_iter):
        if np.any(rhs < -EPS):
            raise ValueError("Получен недопустимый базис")

        rhs[np.abs(rhs) < EPS] = 0.0

        if len(p) == 0 or np.min(p) >= -EPS:
            x = np.zeros(n)
            x[basis] = rhs
            return x, Q, basis

        # Минимальная отрицательная оценка
        j = int(np.argmin(p))

        candidates = [
            i for i in range(len(rhs)) if M[i, j] > EPS
        ]

        if not candidates:
            return None, None, None

        i = min(
            candidates,
            key=lambda k: (
                rhs[k] / M[k, j],
                *tuple(R[k] / M[k, j])
            )
        )

        pivot = M[i, j]
        row = M[i].copy()
        col = M[:, j].copy()
        old_b = rhs[i]
        old_p = p[j]
        old_R = R[i].copy()

        # Метод прямоугольника для остальных элементов
        M -= np.outer(col, row) / pivot

        # Разрешающая строка, столбец и элемент
        M[i] = row / pivot
        M[:, j] = -col / pivot
        M[i, j] = 1.0 / pivot

        # Правые части
        rhs -= col * old_b / pivot
        rhs[i] = old_b / pivot

        # Оценки и значение целевой функции
        p -= old_p * row / pivot
        p[j] = -old_p / pivot
        Q += old_p * old_b / pivot

        # Пересчёт вспомогательной матрицы
        R -= np.outer(col, old_R) / pivot
        R[i] = old_R / pivot

        basis[i], free[j] = free[j], basis[i]

    raise RuntimeError("Превышено число итераций")


def solve_lp(is_max, func_coeffs, table, free_vars=None):
    try:
        n = len(func_coeffs)
        free_vars = sorted(set(free_vars or []))

        if not n or any(j < 0 or j >= n for j in free_vars):
            raise ValueError("Неверные данные о переменных")

        coeffs = list(func_coeffs) + [
            -func_coeffs[j] for j in free_vars
        ]

        expanded = []

        for row, sign, b in table:
            if len(row) != n:
                raise ValueError("Неверное количество коэффициентов")

            new_row = list(row) + [-row[j] for j in free_vars]
            expanded.append((new_row, sign, b))

        rows, b_col, total_vars = to_canonical(
            expanded, len(coeffs)
        )

        A = np.array(rows, dtype=float).reshape(
            len(rows), total_vars
        )
        b = np.array(b_col, dtype=float)

        c = np.array(
            coeffs + [0.0] * (total_vars - len(coeffs)),
            dtype=float
        )

        if not (
            np.all(np.isfinite(A))
            and np.all(np.isfinite(b))
            and np.all(np.isfinite(c))
        ):
            raise ValueError("Коэффициенты должны быть конечными числами")

        if is_max:
            c = -c

        basis = find_basis(A)

        if basis is None:
            # Вспомогательная задача
            m = len(b)
            A_aux = np.hstack([A, np.eye(m)])
            c_aux = np.r_[np.zeros(total_vars), np.ones(m)]
            basis_aux = list(range(total_vars, total_vars + m))

            _, W_aux, basis = simplex(
                c_aux, A_aux, b, basis_aux
            )

            if W_aux is None:
                raise ValueError(
                    "Не удалось решить вспомогательную задачу"
                )

            if W_aux > EPS:
                return None, None, "Область допустимых решений пуста"

            T, rhs = prepare_basis(A_aux, b, basis)

            # Выводим искусственные переменные из базиса
            i = 0

            while i < len(basis):
                if basis[i] < total_vars:
                    i += 1
                    continue

                if abs(rhs[i]) > EPS:
                    return None, None, "Область допустимых решений пуста"

                rhs[i] = 0.0

                j = next(
                    (
                        j for j in range(total_vars)
                        if j not in basis and abs(T[i, j]) > EPS
                    ),
                    None
                )

                if j is None:
                    T = np.delete(T, i, axis=0)
                    rhs = np.delete(rhs, i)
                    basis.pop(i)
                    continue

                pivot = T[i, j]
                T[i] /= pivot
                rhs[i] /= pivot

                for k in range(len(basis)):
                    if k != i:
                        factor = T[k, j]
                        T[k] -= factor * T[i]
                        rhs[k] -= factor * rhs[i]

                basis[i] = j
                i += 1

            # Удаляем искусственные столбцы
            A = T[:, :total_vars]
            b = rhs

        # Основная задача
        x, W, _ = simplex(c, A, b, basis)

        if x is None:
            return None, None, "Целевая функция не ограничена"

        # Восстанавливаем исходные переменные
        result = x[:n].copy()

        for k, j in enumerate(free_vars):
            result[j] -= x[n + k]

        result[np.abs(result) < EPS] = 0.0
        Z = -W if is_max else W

        return result, Z, None

    except (ValueError, RuntimeError) as e:
        return None, None, str(e)


try:
    is_max, func_coeffs, table, free_vars = read_data('data.txt')

    x, Z, err = solve_lp(
        is_max, func_coeffs, table, free_vars
    )

    if err:
        print("Ошибка:", err)
    else:
        print(f"Z = {Z:.6g}")
        for i, value in enumerate(x, start=1):
            print(f"x{i} = {value:.6g}")

except (OSError, ValueError) as e:
    print("Ошибка:", e)
