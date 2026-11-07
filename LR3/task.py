import math
import sys
from itertools import combinations
import sympy as sp

EPS = 1e-8

def solve_system(equations, unknowns):
    equations = [sp.simplify(e) for e in equations if sp.simplify(e) != 0]
    if any(not e.free_symbols and e.is_zero is False for e in equations):
        return []
    if not equations:
        return [{}]
    try:
        result = sp.solve(equations, unknowns, dict=True)
        if result:
            if not isinstance(result, list) or any(not isinstance(s, dict) for s in result):
                raise ValueError("Неожиданный формат ответа solve.")
            return result
    except (NotImplementedError, ValueError, TypeError):
        pass
    try:
        result = sp.nonlinsolve(equations, unknowns)
    except Exception as error:
        raise ValueError(f"Не удалось решить систему уравнений: {error}") from error
    if result == sp.EmptySet:
        return []
    if not isinstance(result, sp.FiniteSet):
        raise ValueError(f"Решатель оставил неразрешённое множество: {result}")
    return [dict(zip(unknowns, values)) for values in result]

def check_residuals(expressions, solution):
    checks = []
    approximate = any(sp.sympify(v).has(sp.Float) for v in solution.values())
    for expression in expressions:
        r = sp.simplify(expression.subs(solution, simultaneous=True))
        if r.free_symbols or r.is_real is not True or r.is_finite is not True:
            raise ValueError(f"Нельзя проверить невязку {r}.")
        error = float(sp.N(abs(r), 20))
        if not math.isfinite(error):
            raise ValueError("Невязка не представима конечным числом.")
        numeric = approximate or expression.has(sp.Float)
        checks.append({"exact": r, "numeric": error, "ok": error <= EPS if numeric else r == 0})
    return checks

def same_point(first, second):
    return all(sp.simplify(a - b) == 0 for a, b in zip(first, second))

def collect_candidates(solutions, variables, f, constraints, multipliers=(), derivatives=()):
    candidates, notes = [], []
    for solution in solutions:
        values = tuple(sp.simplify(sp.sympify(solution.get(v, v)).subs(solution)) for v in variables)
        if any(not isinstance(v, sp.Expr) for v in values):
            notes.append(f"Решатель вернул неявное множество вместо координат: {values}.")
            continue
        if any(v.is_real is False or v.is_finite is False for v in values):
            continue
        if any(v.free_symbols for v in values):
            notes.append(f"Параметрическое семейство {values} требует отдельного исследования.")
            continue
        if any(v.is_real is not True or v.is_finite is not True for v in values):
            notes.append(f"Действительность решения {values} не установлена.")
            continue
        point = dict(zip(variables, values))
        residuals = check_residuals(constraints, point)
        if not all(r["ok"] for r in residuals):
            continue
        lam_values = None
        if multipliers:
            params = {lam: 0 for lam in multipliers if lam not in solution}
            lam_values = tuple(sp.simplify(sp.sympify(solution.get(lam, lam)).subs(point).subs(params))
                               for lam in multipliers)
            if any(v.free_symbols or v.is_real is not True or v.is_finite is not True
                   for v in lam_values):
                notes.append(f"Не удалось определить множители в точке {values}.")
                continue
            substitution = dict(point)
            substitution.update(zip(multipliers, lam_values))
            stationarity = check_residuals(derivatives, substitution)
            if not all(r["ok"] for r in stationarity):
                continue
        else:
            stationarity = []
        value = sp.simplify(f.subs(point, simultaneous=True))
        if value.is_real is not True or value.is_finite is not True:
            notes.append(f"Не удалось вычислить конечное действительное f в {values}.")
            continue
        candidates.append({"point": values, "solution": point, "multipliers": lam_values,
                           "value": value, "constraints": residuals,
                           "stationarity": stationarity, "singular": not bool(multipliers)})
    return candidates, notes

def find_singular_points(variables, constraints, jacobian):
    m, n = len(constraints), len(variables)
    if m == 0:
        return []
    minors = [sp.simplify(jacobian[:, columns].det())
              for columns in combinations(range(n), m)] if m <= n else []
    if any(not minor.free_symbols and minor.is_zero is False for minor in minors):
        return []
    return solve_system(list(constraints) + minors, variables)

def feasible_paths(variables, f, constraints):
    paths = []
    name = "t"
    while name in {v.name for v in variables}:
        name += "1"
    parameter = sp.Symbol(name, real=True)
    affine = all(sp.Poly(h, *variables).total_degree() <= 1 for h in constraints)
    if affine:
        solutions = solve_system(constraints, tuple(reversed(variables)))
        if not solutions:
            return paths
        mapping = {v: sp.simplify(solutions[0].get(v, v)) for v in variables}
        free = sorted(set().union(*(e.free_symbols for e in mapping.values())), key=sp.default_sort_key)
        directions = [{v: int(v == u) for v in free} for u in free]
        for u, v in combinations(free, 2):
            for sign in (-1, 1):
                directions.append({w: (1 if w == u else sign if w == v else 0) for w in free})
        for direction in directions:
            substitutions = {v: direction[v] * parameter for v in free}
            paths.append(tuple(sp.simplify(mapping[v].subs(substitutions, simultaneous=True))
                               for v in variables))
    elif len(variables) == 2 and len(constraints) == 1:
        for free, dependent in (variables, tuple(reversed(variables))):
            for solution in solve_system(constraints, (dependent,)):
                expression = solution.get(dependent)
                if not isinstance(expression, sp.Expr) or expression.free_symbols - {free}:
                    continue
                if expression.is_rational_function(free) is not True:
                    continue
                mapping = {free: parameter, dependent: expression.subs(free, parameter)}
                paths.append(tuple(mapping[v] for v in variables))
    checked = []
    for values in paths:
        real_path = True
        for coordinate in values:
            numerator, denominator = sp.cancel(coordinate).as_numer_denom()
            if coordinate.is_rational_function(parameter) is not True or denominator == 0:
                real_path = False
                break
            coefficients = sp.Poly(numerator, parameter).coeffs() + sp.Poly(denominator, parameter).coeffs()
            if any(c.is_real is not True or c.is_finite is not True for c in coefficients):
                real_path = False
                break
        if not real_path:
            continue
        mapping = dict(zip(variables, values))
        if any(sp.simplify(h.subs(mapping, simultaneous=True)) != 0 for h in constraints):
            continue
        phi = sp.factor(f.subs(mapping, simultaneous=True))
        try:
            limits = (sp.limit(phi, parameter, -sp.oo), sp.limit(phi, parameter, sp.oo))
        except (NotImplementedError, ValueError):
            continue
        checked.append({"point": values, "parameter": parameter, "phi": phi, "limits": limits})
    return checked

def classify(variables, f, constraints, multipliers, L, jacobian, candidates):
    for candidate in candidates:
        candidate["global_min"] = candidate["global_max"] = False
        candidate["classification"] = "особая точка; требуется отдельная классификация"
        if candidate["multipliers"] is None:
            continue
        substitution = dict(candidate["solution"])
        substitution.update(zip(multipliers, candidate["multipliers"]))
        fixed_L = sp.simplify(L.subs(dict(zip(multipliers, candidate["multipliers"]))))
        H = sp.simplify(sp.hessian(L, variables).subs(substitution))
        if sp.Poly(fixed_L, *variables).total_degree() <= 2:
            candidate["global_min"] = H.is_positive_semidefinite is True
            candidate["global_max"] = H.is_negative_semidefinite is True
        J = jacobian.subs(candidate["solution"])
        candidate["singular"] = J.rank() < len(constraints)
        if not candidate["singular"]:
            basis = J.nullspace() if constraints else [sp.eye(len(variables))[:, i] for i in range(len(variables))]
            if not basis:
                candidate["classification"] = "изолированная допустимая точка"
            else:
                Z = sp.Matrix.hstack(*basis)
                restricted = sp.simplify(Z.T * H * Z)
                if restricted.is_positive_definite is True:
                    candidate["classification"] = "строгий локальный минимум"
                elif restricted.is_negative_definite is True:
                    candidate["classification"] = "строгий локальный максимум"
                elif (restricted.is_positive_semidefinite is False
                      and restricted.is_negative_semidefinite is False):
                    candidate["classification"] = "седловая точка"
                else:
                    candidate["classification"] = "второй порядок не даёт окончательной классификации"
        if candidate["global_min"]:
            candidate["classification"] = "глобальный минимум"
        if candidate["global_max"]:
            candidate["classification"] = "глобальный максимум"

    reasons = []
    extrema = {}
    for name, flag in (("minimum", "global_min"), ("maximum", "global_max")):
        certified = [c for c in candidates if c[flag]]
        if certified:
            value = certified[0]["value"]
            if any(sp.simplify(c["value"] - value) != 0 for c in certified):
                raise ValueError("Противоречивые сертификаты глобальности.")
            points = [c["point"] for c in candidates if sp.simplify(c["value"] - value) == 0]
            extrema[name] = {"status": "exists", "value": value, "points": points}
            sign = "положительно" if name == "minimum" else "отрицательно"
            reasons.append(f"Гессиан квадратичного лагранжиана при найденных множителях {sign} полуопределён: "
                           f"это доказывает глобальность {'минимума' if name == 'minimum' else 'максимума'}.")
        else:
            extrema[name] = {"status": "unknown", "points": []}

    singleton = None
    for h in constraints:
        if sp.Poly(h, *variables).total_degree() != 2:
            continue
        Q = sp.hessian(h, variables) / 2
        if Q.is_positive_definite is not True and Q.is_negative_definite is not True:
            continue
        zero = dict.fromkeys(variables, 0)
        linear = sp.Matrix([sp.diff(h, v).subs(zero) for v in variables])
        center = tuple(sp.simplify(v) for v in -Q.inv() * linear / 2)
        substitution = dict(zip(variables, center))
        if all(sp.simplify(g.subs(substitution)) == 0 for g in constraints):
            singleton = center
            value = sp.simplify(f.subs(substitution))
            for name in ("minimum", "maximum"):
                extrema[name] = {"status": "exists", "value": value, "points": [center]}
            for c in candidates:
                c["classification"] = "глобальный минимум и глобальный максимум"
            reasons.append("Определённое квадратичное ограничение имеет нулевой уровень только в своём центре; "
                           "это единственная допустимая точка, поэтому она даёт минимум и максимум.")
            break
    if singleton is None and any(c["singular"] for c in candidates):
        feasible = solve_system(constraints, variables)
        finite, notes = collect_candidates(feasible, variables, f, constraints)
        if finite and not notes:
            for name, order in (("minimum", -1), ("maximum", 1)):
                best = finite[0]["value"]
                for c in finite[1:]:
                    difference = sp.simplify(c["value"] - best)
                    if (difference * order).is_positive is True:
                        best = c["value"]
                    elif difference != 0 and difference.is_positive is not True and difference.is_negative is not True:
                        raise ValueError("Не удалось точно сравнить значения на конечном множестве.")
                extrema[name] = {"status": "exists", "value": best,
                                 "points": [c["point"] for c in finite if sp.simplify(c["value"] - best) == 0]}
            reasons.append("Все допустимые точки найдены; глобальные экстремумы установлены сравнением их значений.")
            for c in candidates:
                kinds = []
                if sp.simplify(c["value"] - extrema["minimum"]["value"]) == 0:
                    kinds.append("глобальный минимум")
                if sp.simplify(c["value"] - extrema["maximum"]["value"]) == 0:
                    kinds.append("глобальный максимум")
                c["classification"] = " и ".join(kinds) or c["classification"]

    paths = feasible_paths(variables, f, constraints)
    for name, infinity in (("minimum", -sp.oo), ("maximum", sp.oo)):
        path = next((p for p in paths if infinity in p["limits"]), None)
        if path is None:
            continue
        if extrema[name]["status"] == "exists":
            raise ValueError("Путь на бесконечности противоречит найденному экстремуму.")
        extrema[name] = {"status": "absent", "points": []}
        reasons.append(f"Вдоль допустимого пути {path['point']} при достаточно больших |t| "
                       f"функция равна {path['phi']}; "
                       f"пределы {path['limits']}, поэтому {'минимума' if name == 'minimum' else 'максимума'} нет.")
    return extrema, reasons, paths

def solve_lab(variables, f, constraints):
    variables = tuple(variables)
    f, constraints = sp.sympify(f), [sp.sympify(h) for h in constraints]
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("Нужен непустой список различных переменных.")
    if any(not isinstance(v, sp.Symbol) or v.is_real is not True for v in variables):
        raise ValueError("Создайте переменные с real=True.")
    if not isinstance(EPS, (int, float)) or not math.isfinite(EPS) or EPS <= 0:
        raise ValueError("EPS должен быть конечным положительным числом.")
    for expression in [f] + constraints:
        if expression.free_symbols - set(variables) or not expression.is_polynomial(*variables):
            raise ValueError("Эта версия работает с многочленами от заданных переменных.")
        if any(c.is_real is not True or c.is_finite is not True for c in sp.Poly(expression, *variables).coeffs()):
            raise ValueError("Коэффициенты должны быть конечными действительными числами.")
    multipliers = tuple(sp.symbols(f"lambda1:{len(constraints)+1}", real=True))
    L = f + sum(lam * h for lam, h in zip(multipliers, constraints))
    derivatives = [sp.diff(L, v) for v in variables]
    equations = derivatives + constraints
    unknowns = variables + multipliers
    solutions = solve_system(equations, unknowns)
    candidates, notes = collect_candidates(solutions, variables, f, constraints, multipliers, derivatives)
    J = sp.Matrix(constraints).jacobian(variables) if constraints else sp.zeros(0, len(variables))
    singular_solutions = find_singular_points(variables, constraints, J)
    singular, singular_notes = collect_candidates(singular_solutions, variables, f, constraints)
    notes += singular_notes
    for candidate in singular:
        existing = next((c for c in candidates if same_point(c["point"], candidate["point"])), None)
        if existing is None:
            candidates.append(candidate)
        else:
            existing["singular"] = True
    candidates.sort(key=lambda c: sp.default_sort_key(sp.Tuple(*c["point"])))
    extrema, reasons, paths = classify(variables, f, constraints, multipliers, L, J, candidates)
    if not candidates and not paths:
        feasible = solve_system(constraints, variables)
        real_feasible, unresolved = collect_candidates(feasible, variables, f, constraints)
        if not real_feasible and not unresolved:
            raise ValueError("Допустимое множество пусто: действительных решений ограничений нет.")
    reasons += notes
    for i, candidate in enumerate(candidates, 1):
        reasons.append(f"Кандидат {i}: {candidate['classification']}.")
    if any(extrema[name]["status"] == "unknown" for name in extrema):
        reasons.append("Для части выводов требуется дополнительное математическое исследование.")
    return {"variables": variables, "f": f, "constraints": constraints, "multipliers": multipliers,
            "L": L, "derivatives": derivatives, "equations": equations, "jacobian": J,
            "candidates": candidates, "notes": notes, "extrema": extrema,
            "reason": " ".join(reasons), "paths": paths, "singular_solutions": singular_solutions}


def show_extremum(extremum):
    if extremum["status"] == "absent":
        return "не существует"
    if extremum["status"] == "unknown":
        return "не установлен"
    label = "в точке" if len(extremum["points"]) == 1 else "в точках"
    points = ", ".join(str(p) for p in extremum["points"])
    return f"{extremum['value']} {label} {points}"

def print_result(result):
    print("Целевая функция:", result["f"])
    print("Ограничения:", "; ".join(f"{h} = 0" for h in result["constraints"]) or "нет")
    print("Функция Лагранжа:", result["L"])
    print("Уравнения стационарности:")
    for equation in result["derivatives"]:
        print(" ", equation, "= 0")
    for i, candidate in enumerate(result["candidates"], 1):
        print(f"\nКандидат {i}:")
        print("  точка:", candidate["point"])
        print("  множители:", candidate["multipliers"] if candidate["multipliers"] is not None else "не существуют или не определены (особая точка)")
        print("  значение функции:", candidate["value"])
        print("  невязки ограничений:", [r["exact"] for r in candidate["constraints"]])
        print("  допустимость:", "да" if all(r["ok"] for r in candidate["constraints"]) else "нет")
    print("\nИтог:")
    print("  глобальный минимум:", show_extremum(result["extrema"]["minimum"]))
    print("  глобальный максимум:", show_extremum(result["extrema"]["maximum"]))
    print("  основание классификации:", result["reason"])


def main():
    try:
        x, y, z = sp.symbols("x y z", real=True)
        variables = (x, y, z)
        f = x**2 + y**2 + z**2
        constraints = [x + y + z - 3, x - y]
        result = solve_lab(variables, f, constraints)
        print_result(result)
        return 0
    except ValueError as error:
        print("Ошибка:", error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Вычисления прерваны.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Ошибка вычислений ({type(error).__name__}): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
