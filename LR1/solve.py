import numpy as np

np.set_printoptions(precision=4, suppress=True)

def read_data(file_name):
    with open(file_name, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    is_max = lines[0] == 'max'
    func_coeffs = list(map(float, lines[1].split()))

    table = []
    for line in lines[2:]:
        raw = line.split()
        b = float(raw[-1])
        sign = raw[-2]
        if sign not in ('=', '>=', '<='):
            raise ValueError("знак должен быть один из: =, >=, <=")
        coeffs = list(map(float, raw[:-2]))
        table.append((coeffs, sign, b))
    return is_max, func_coeffs, table

def to_canonical(table, n_vars):
    # Приводит систему к каноническому виду, добавляя свободные переменные
    rows = []
    b_col = []
    extra = 0
    for coeffs, sign, b in table:
        row = list(coeffs) + [0.0] * 20
        if sign == '=':
            if b < 0:
                row = [-v for v in row]
                b = -b
        elif sign == '<=':
            row[n_vars + extra] = 1.0
            extra += 1
        elif sign == '>=':
            # Добавляем избыточную переменную со знаком -1
            row[n_vars + extra] = -1.0
            extra += 1
            # Если правая часть отрицательная значит умножаем всё ограничение на -1, и при этом знак избыточной переменной меняется на +1, что позволяет нам
            # использовать её как базисную во вспомогательной задаче.
            if b < 0:
                row = [-v for v in row]
                b = -b
        rows.append(row)
        b_col.append(b)

    total_vars = n_vars + extra
    rows = [r[:total_vars] for r in rows]
    return rows, b_col, total_vars

def has_identity_basis(A):
    # Проверяет, есть ли в матрице A единичный базис
    m, n = A.shape
    used = set()
    for i in range(m):
        found = -1
        for j in range(n):
            if j in used:
                continue
            if abs(A[i, j] - 1.0) < 1e-9 and all(
                abs(A[k, j]) < 1e-9 for k in range(m) if k != i
            ):
                found = j
                break
        if found == -1:
            return False
        used.add(found)
    return True


def find_basis(A):
    # Возвращает список индексов базисных столбцов
    m, n = A.shape
    basis = []
    used = set()
    for i in range(m):
        for j in range(n):
            if j in used:
                continue
            if abs(A[i, j] - 1.0) < 1e-9 and all(
                abs(A[k, j]) < 1e-9 for k in range(m) if k != i
            ):
                basis.append(j)
                used.add(j)
                break
    return basis


def simplex(c, A, b, basis, max_iter=100):
    m, n = A.shape
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    basis = list(basis)

    for _ in range(max_iter):
        B = A[:, basis]
        xB = np.linalg.solve(B, b)
        cB = c[basis]
        y = np.linalg.solve(B.T, cB)
        reduced = c - A.T.dot(y)
        # Выбор разрешающего столбца
        entering = None
        for j in range(n):
            if j not in basis and reduced[j] < -1e-9:
                entering = j
                break

        if entering is None:
            x = np.zeros(n)
            x[basis] = xB
            value = 0.0
            for j in range(n):
                value += c[j] * x[j]
            return x, value, basis

        # Выбор разрешающей строки
        d = np.linalg.solve(B, A[:, entering])
        ratios = []
        for i in range(m):
            if d[i] > 1e-9:
                ratios.append((xB[i] / d[i], i))
        if not ratios:
            return None, None, None

        i, leaving_idx = min(ratios)
        pivot = A[leaving_idx, entering]

        A[leaving_idx, :] = A[leaving_idx, :] / pivot
        b[leaving_idx] = b[leaving_idx] / pivot

        # Обнуляем разрешающий столбец в остальных строках
        for i in range(m):
            if i != leaving_idx:
                factor = A[i, entering]
                A[i, :] = A[i, :] - factor * A[leaving_idx, :]
                b[i] = b[i] - factor * b[leaving_idx]

        # Меняем базисную переменную
        basis[leaving_idx] = entering
    return None, None, None

def solve_lp(is_max, func_coeffs, table):
    n = len(func_coeffs)

    # Канонический вид
    rows, b_col, total_vars = to_canonical(table, n)
    A = np.array(rows, dtype=float)
    b = np.array(b_col, dtype=float)

    # Если нет единичного базиса — вспомогательная задача
    if not has_identity_basis(A):
        m = A.shape[0]
        A_aux = np.hstack([A, np.eye(m)])
        c_aux = np.zeros(total_vars + m)
        for i in range(m):
            c_aux[total_vars + i] = 1.0
        basis_aux = [total_vars + i for i in range(m)]

        x_aux, W_aux, basis_aux = simplex(c_aux, A_aux, b, basis_aux)

        if W_aux is None or W_aux > 1e-6:
            return None, None, "Область допустимых решений пуста"

        A = A_aux[:, :total_vars]
        basis = [j for j in basis_aux if j < total_vars]
    else:
        basis = find_basis(A)

    # Основная задача
    if is_max:
        c = [-v for v in func_coeffs] + [0.0] * (total_vars - n)
    else:
        c = [v for v in func_coeffs] + [0.0] * (total_vars - n)

    x, W, basis = simplex(c, A, b, basis)

    if x is None:
        return None, None, "Решение не ограничено"

    Z = -W if is_max else W
    return x[:n], Z, None

is_max, func_coeffs, table = read_data('data.txt')

x, Z, err = solve_lp(is_max, func_coeffs, table)

if err:
    print("Ошибка:", err)
else:
    print(f"Z = {Z:.1f}")
    for i in range(len(x)):
        print(f"x{i+1} = {x[i]}")