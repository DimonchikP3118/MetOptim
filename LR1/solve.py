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
        # Правая часть каждого равенства должна быть неотрицательной
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
                abs(A[k, j]) < EPS for k in range(m) if k != i):
                found = j
                break
        if found is None:
            return None
        basis.append(found)
        used.add(found)
    return basis

def has_identity_basis(A):
    return find_basis(A) is not None

def simplex(c, A, b, basis, max_iter=10000):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    basis = list(basis)
    m, n = A.shape
    for _ in range(max_iter):
        B = A[:, basis]
        xB = np.linalg.solve(B, b)
        if np.any(xB < -EPS):
            raise ValueError("Получен недопустимый базис")
        xB[np.abs(xB) < EPS] = 0.0
        cB = c[basis]
        y = np.linalg.solve(B.T, cB)
        reduced = c - A.T @ y
        # Выбираем переменную с наименьшим подходящим индексом
        entering = next(
            (j for j in range(n)
                if j not in basis and reduced[j] < -EPS),None)
        if entering is None:
            x = np.zeros(n)
            x[basis] = xB
            return x, float(c @ x), basis

        d = np.linalg.solve(B, A[:, entering])

        candidates = [i for i in range(m) if d[i] > EPS]
        if not candidates:
            return None, None, None

        min_ratio = min(xB[i] / d[i] for i in candidates)
        tied = [
            i for i in candidates
            if abs(xB[i] / d[i] - min_ratio) <= EPS
        ]

        leaving_idx = min(tied, key=lambda i: basis[i])
        basis[leaving_idx] = entering
    raise RuntimeError("Превышено число итераций")

def solve_lp(is_max, func_coeffs, table, free_vars=None):
    try:
        n = len(func_coeffs)
        free_vars = sorted(set(free_vars or []))
        if not n or any(j < 0 or j >= n for j in free_vars):
            raise ValueError("Неверные данные о переменных")
        coeffs = list(func_coeffs) + [
            -func_coeffs[j] for j in free_vars]
        expanded = []

        for row, sign, b in table:
            if len(row) != n:
                raise ValueError("Неверное количество коэффициентов")
            new_row = list(row) + [-row[j] for j in free_vars]
            expanded.append((new_row, sign, b))

        rows, b_col, total_vars = to_canonical(
            expanded, len(coeffs))

        A = np.array(rows, dtype=float).reshape(
            len(rows), total_vars)
        b = np.array(b_col, dtype=float)
        c = np.array(
            coeffs + [0.0] * (total_vars - len(coeffs)),
            dtype=float)

        if not (
            np.all(np.isfinite(A))
            and np.all(np.isfinite(b))
            and np.all(np.isfinite(c))):
            raise ValueError("Коэффициенты должны быть конечными числами")

        if is_max:
            c = -c
        if not has_identity_basis(A):
            m = len(b)
            A_aux = np.hstack([A, np.eye(m)])
            c_aux = np.r_[np.zeros(total_vars), np.ones(m)]
            basis_aux = list(range(total_vars, total_vars + m))
            _, W_aux, basis = simplex(
                c_aux, A_aux, b, basis_aux)

            if W_aux is None:
                raise ValueError(
                    "Не удалось решить вспомогательную задачу")
            if W_aux > EPS:
                return None, None, "Область допустимых решений пуста"
            # Приводим систему к найденному базису
            B = A_aux[:, basis]
            T = np.linalg.solve(B, A_aux)
            rhs = np.linalg.solve(B, b)

            i = 0

            while i < len(basis):
                if basis[i] < total_vars:
                    i += 1
                    continue
                if abs(rhs[i]) > EPS:
                    return None, None, "Область допустимых решений пуста"
                rhs[i] = 0.0

                j = next(
                    (j for j in range(total_vars)
                        if j not in basis and abs(T[i, j]) > EPS),None)
                if j is None:
                    T = np.delete(T, i, axis=0)
                    rhs = np.delete(rhs, i)
                    basis.pop(i)
                    continue
                # Выводим искусственную переменную из базиса
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
            # Удаляем искусственные столбцы после очистки базиса
            A = T[:, :total_vars]
            b = rhs
        else:
            basis = find_basis(A)
        x, W, _ = simplex(c, A, b, basis)
        if x is None:
            return None, None, "Целевая функция не ограничена"
        # Восстанавливаем исходные свободные переменные
        result = x[:n].copy()

        for k, j in enumerate(free_vars):
            result[j] -= x[n + k]

        result[np.abs(result) < EPS] = 0.0
        Z = -W if is_max else W
        return result, Z, None

    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        return None, None, str(e)

try:
    is_max, func_coeffs, table, free_vars = read_data('data.txt')

    x, Z, err = solve_lp(
        is_max, func_coeffs, table, free_vars)

    if err:
        print("Ошибка:", err)
    else:
        print(f"Z = {Z:.6g}")
        for i, value in enumerate(x, start=1):
            print(f"x{i} = {value:.6g}")

except (OSError, ValueError) as e:
    print("Ошибка:", e)
