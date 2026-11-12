from fractions import Fraction
from decimal import Decimal, localcontext
from itertools import product
from math import gcd
from pathlib import Path
from time import monotonic
import csv
import sys

DATA = {
    "names": ("ЦБ1", "ЦБ2", "Депозит"),
    "initial": (100, 800, 400),
    "cash": 600,
    "minimum": (30, 150, 100),
    "lots": None,
    "commissions": ("0.04", "0.07", "0.05"),
    "stages": (
        (("Благоприятная", "0.6", ("1.20", "1.10", "1.07")),
         ("Нейтральная", "0.3", ("1.05", "1.02", "1.03")),
         ("Негативная", "0.1", ("0.80", "0.95", "1.00"))),
        (("Благоприятная", "0.3", ("1.40", "1.15", "1.01")),
         ("Нейтральная", "0.2", ("1.05", "1.00", "1.00")),
         ("Негативная", "0.5", ("0.60", "0.90", "1.00"))),
        (("Благоприятная", "0.4", ("1.15", "1.12", "1.05")),
         ("Нейтральная", "0.4", ("1.05", "1.01", "1.01")),
         ("Негативная", "0.2", ("0.70", "0.94", "1.00"))),),}

LIMITS = {"max_states": 100_000, "max_checks": 5_000_000, "max_seconds": 120}
USE_BOUNDS = True

def number(value, label):
    if isinstance(value, bool) or isinstance(value, complex):
        raise ValueError(f"{label}: нужно конечное действительное число.")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{label}: неверное число {value!r}.") from None
    return result

def sequence(value, label, size=None):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label}: нужен список или кортеж.")
    if size is not None and len(value) != size:
        raise ValueError(f"{label}: нужно элементов {size}, получено {len(value)}.")
    return value

def common_denominator(values):
    result = 1
    for v in values:
        result = result * v.denominator // gcd(result, v.denominator)
    return result

def ceil_div(a, b):
    return -((-a) // b)

def text(value):
    value = Fraction(value)
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        return str(value)
    with localcontext() as context:
        context.prec = len(str(value.numerator)) + len(str(value.denominator)) + 2
        result = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return result.rstrip("0").rstrip(".") if "." in result else result

def fail_if(condition, message):
    if condition:
        raise ValueError(message)

class PortfolioDP:
    def __init__(self, data, limits=None, use_bounds=True):
        fail_if(limits is not None and (not isinstance(limits, dict)), "Лимиты должны быть словарём.")
        fail_if(type(use_bounds) is not bool, "use_bounds должен быть True или False.")
        self.limits = dict(LIMITS if limits is None else limits)
        fail_if(set(self.limits) != {'max_states', 'max_checks', 'max_seconds'}, "В LIMITS нужны поля max_states, max_checks и max_seconds.")
        for key in ("max_states", "max_checks"):
            fail_if(type(self.limits.get(key)) is not int or self.limits[key] <= 0, f"Лимит {key} должен быть положительным целым.")
        seconds = self.limits.get("max_seconds")
        fail_if(seconds is not None and number(seconds, 'Лимит времени') <= 0, "Лимит времени должен быть положительным или None.")
        self.max_seconds = None if seconds is None else number(seconds, "Лимит времени")
        self.use_bounds = bool(use_bounds)
        self.prepare(data)
        self.values, self.policy, self.tables = {}, {}, {}
        self.stats = {"states": 0, "candidate_checks": 0}
        self.started = None
        self.result = None
        self.finished = False

    def prepare(self, data):
        fail_if(not isinstance(data, dict), "Исходные данные должны быть словарём.")
        fields = {"names", "initial", "cash", "minimum", "lots", "commissions", "stages"}
        fail_if(set(data) != fields, f"Неверные поля данных. Нужны: {', '.join(sorted(fields))}.")
        self.names = tuple(sequence(data["names"], "Названия активов"))
        fail_if(any((not isinstance(s, str) or not s.strip() for s in self.names)), "Названия активов должны быть непустыми строками.")
        fail_if(len(set(self.names)) != len(self.names), "Названия активов не должны повторяться.")
        self.n = len(self.names)

        def vector(key):
            return tuple(number(v, f"{key}[{i+1}]") for i, v in enumerate(sequence(data[key], key, self.n)))

        initial, minimum, commissions = vector("initial"), vector("minimum"), vector("commissions")
        cash = number(data["cash"], "Свободные средства")
        fail_if(any((x < 0 for x in initial + minimum + (cash,))), "Начальные средства и минимумы не могут быть отрицательными.")
        fail_if(any((x < 0 or x > 1 for x in commissions)), "Комиссии задаются долями от 0 до 1 включительно.")
        lots = tuple(x / 4 for x in initial) if data["lots"] is None else vector("lots")
        fail_if(any((q <= 0 for q in lots)), "Пакет должен быть положительным. Для нулевого начального актива задайте lots явно.")
        fees = tuple(q * f for q, f in zip(lots, commissions))
        self.base = common_denominator(initial + minimum + lots + fees + (cash,))
        units = lambda v: int(v * self.base)
        self.initial = tuple(map(units, initial + (cash,)))
        self.minimum = tuple(map(units, minimum))
        self.lots = tuple(map(units, lots))
        self.fees = tuple(map(units, fees))
        self.initial_total = sum(initial) + cash
        self.stages = []
        for k, raw_stage in enumerate(sequence(data["stages"], "Этапы"), 1):
            raw_stage = sequence(raw_stage, f"Этап {k}")
            fail_if(not raw_stage, f"Этап {k}: нет сценариев.")
            stage, labels = [], set()
            for j, row in enumerate(raw_stage, 1):
                label, p, rates = sequence(row, f"Сценарий {k}.{j}", 3)
                fail_if(not isinstance(label, str) or not label.strip() or label in labels, f"Этап {k}: названия сценариев должны быть непустыми и различными.")
                labels.add(label)
                p = number(p, f"Вероятность {k}.{j}")
                rates = tuple(number(r, f"Коэффициент {k}.{j}.{i+1}") for i, r in enumerate(
                    sequence(rates, f"Коэффициенты {k}.{j}", self.n)))
                fail_if(p < 0 or p > 1 or any((r < 0 for r in rates)), f"Сценарий {k}.{j}: вероятность должна быть от 0 до 1, коэффициенты — неотрицательными.")
                stage.append((label, p, rates))
            fail_if(sum((row[1] for row in stage)) != 1, f"Этап {k}: сумма вероятностей должна быть точно 1. Для точных долей можно писать '1/3'.")
            self.stages.append([row for row in stage if row[1] > 0])
        self.N = len(self.stages)
        if self.N + 100 >= sys.getrecursionlimit():
            raise RuntimeError("Слишком много этапов для рекурсивной реализации.")
        self.P, self.R, self.probs, self.rates, self.means = [], [], [], [], []
        for stage in self.stages:
            pd = common_denominator([row[1] for row in stage])
            rd = common_denominator([r for row in stage for r in row[2]])
            ps = [int(row[1] * pd) for row in stage]
            rs = [tuple(int(r * rd) for r in row[2]) for row in stage]
            self.P.append(pd)
            self.R.append(rd)
            self.probs.append(ps)
            self.rates.append(rs)
            self.means.append(tuple(sum(p * row[i] for p, row in zip(ps, rs))
                                    for i in range(self.n)))
        self.factor = [1]
        for rd in self.R:
            self.factor.append(self.factor[-1] * rd)
        self.value_scale = [self.base * self.factor[-1]] * (self.N + 1)
        self.growth = [1] * (self.N + 1)
        for k in range(self.N-1, -1, -1):
            self.value_scale[k] = self.value_scale[k+1] * self.P[k]
            one_step = sum(p * max((self.R[k],) + row)
                           for p, row in zip(self.probs[k], self.rates[k]))
            self.growth[k] = self.growth[k+1] * one_step
        self.safe_last = self.N > 0 and all(
            (q-f) * self.R[-1] * self.P[-1] <= q*w <= (q+f) * self.R[-1] * self.P[-1]
            for q, f, w in zip(self.lots, self.fees, self.means[-1]))

    def check_time(self):
        if self.max_seconds is not None and monotonic() - self.started > self.max_seconds:
            raise RuntimeError("Превышен лимит времени. Увеличьте max_seconds или уменьшите задачу.")

    def state_text(self, k, state):
        return "(" + "; ".join(text(Fraction(x, self.base * self.factor[k])) for x in state) + ")"

    def value_text(self, k, value):
        return "" if value is None else text(Fraction(value, self.value_scale[k]))

    def action_range(self, k, state):
        f = self.factor[k]
        low = tuple(ceil_div(m*f-a, q*f) for a, m, q in zip(state[:-1], self.minimum, self.lots))
        available = state[-1] + sum(max(0, -u) * (q-fee) * f
                                    for u, q, fee in zip(low, self.lots, self.fees))
        high = tuple(available // ((q+fee)*f) for q, fee in zip(self.lots, self.fees))
        return low, high

    def trade(self, k, state, action):
        f = self.factor[k]
        assets = tuple(a + q*u*f for a, q, u in zip(state[:-1], self.lots, action))
        cash = state[-1] - sum((q*u+fee*abs(u))*f
                               for q, fee, u in zip(self.lots, self.fees, action))
        return assets + (cash,)

    def feasible(self, k, post):
        return post[-1] >= 0 and all(a >= m*self.factor[k]
                                     for a, m in zip(post[:-1], self.minimum))

    def controls(self, k, state):
        low, high = self.action_range(k, state)
        count = 1
        for lo, hi in zip(low, high):
            count *= max(0, hi-lo+1)
        if count == 0:
            return
        if count > self.limits["max_checks"] - self.stats["candidate_checks"]:
            raise RuntimeError("Слишком большой перебор в состоянии. Увеличьте max_checks или размеры пакетов.")
        for action in product(*(range(a, b+1) for a, b in zip(low, high))):
            self.stats["candidate_checks"] += 1
            if self.stats["candidate_checks"] > self.limits["max_checks"]:
                raise RuntimeError("Превышен лимит перебора. Увеличьте max_checks или размеры пакетов.")
            if self.stats["candidate_checks"] % 1024 == 0:
                self.check_time()
            post = self.trade(k, state, action)
            if self.feasible(k, post):
                yield action, post

    def transition(self, k, post, j):
        return tuple(a*r for a, r in zip(post[:-1], self.rates[k][j])) + (post[-1]*self.R[k],)

    def last_linear(self, state):
        return sum(a*w for a, w in zip(state[:-1], self.means[-1])) + state[-1]*self.R[-1]*self.P[-1]

    def penultimate_bound(self, state):
        k = self.N - 2
        low, high = self.action_range(k, state)
        weights = tuple(a*b for a, b in zip(self.means[k], self.means[k+1]))
        cash_weight = self.R[k]*self.P[k]*self.R[k+1]*self.P[k+1]
        result = state[-1]*cash_weight
        for a, q, fee, w, lo, hi in zip(state[:-1], self.lots, self.fees, weights, low, high):
            if lo > hi:
                return None
            candidates = {lo, hi}
            if lo <= 0 <= hi:
                candidates.add(0)
            result += a*w + self.factor[k]*max(q*u*w-cash_weight*(q*u+fee*abs(u)) for u in candidates)
        return result

    def action_bound(self, k, post):
        if k == self.N-1:
            return self.last_linear(post)
        total = 0
        for j, p in enumerate(self.probs[k]):
            state = self.transition(k, post, j)
            if self.safe_last and k == self.N-2:
                bound = self.last_linear(state)
            elif self.safe_last and k == self.N-3:
                bound = self.penultimate_bound(state)
                if bound is None:
                    return None
            else:
                bound = sum(state)*self.growth[k+1]
            total += p*bound
        return total

    def solve(self):
        self.started = monotonic()
        self.result = self.bellman(0, self.initial)
        self.finished = True
        return None if self.result is None else Fraction(self.result, self.value_scale[0])

    def bellman(self, k, state):
        key = (k, state)
        if key in self.values:
            return self.values[key]
        self.check_time()
        self.stats["states"] += 1
        if self.stats["states"] > self.limits["max_states"]:
            raise RuntimeError("Превышен лимит состояний. Увеличьте max_states или уменьшите задачу.")
        if k == self.N:
            self.values[key], self.policy[key], self.tables[key] = sum(state), (), []
            return sum(state)
        ranked, rows = [], []
        for action, post in self.controls(k, state):
            bound = self.action_bound(k, post) if self.use_bounds or k == self.N-1 else None
            row = {"action": action, "upper": bound, "value": None, "status": "ожидает"}
            rows.append(row)
            if bound is None and self.use_bounds:
                row["status"] = "нет допустимого продолжения"
            else:
                ranked.append((bound, action, post, row))
        ranked.sort(key=lambda item: (0 if item[0] is None else -item[0], item[1]))
        best, best_actions = None, []
        for bound, action, post, row in ranked:
            if k != self.N-1 and best is not None and bound is not None and bound < best:
                row["status"] = "отсечено по верхней оценке"
                continue
            if k == self.N-1:
                value = bound
            else:
                value = 0
                for j, p in enumerate(self.probs[k]):
                    child = self.bellman(k+1, self.transition(k, post, j))
                    if child is None:
                        value = None
                        break
                    value += p*child
            row["value"] = value
            row["status"] = "вычислено" if value is not None else "нет допустимого продолжения"
            if value is not None and bound is not None and value > bound:
                raise ArithmeticError("Полученное значение превысило верхнюю оценку.")
            if value is not None:
                if best is None or value > best:
                    best, best_actions = value, [action]
                elif value == best:
                    best_actions.append(action)
        self.values[key], self.policy[key], self.tables[key] = best, tuple(sorted(best_actions)), rows
        return best

    def audit(self):
        if not self.finished or self.result is None:
            raise RuntimeError("Сначала нужно найти допустимую стратегию.")
        current = {self.initial: (Fraction(1), 1)}
        plan = []
        for k in range(self.N):
            following = {}
            for state, (prob, histories) in current.items():
                action = self.policy[k, state][0]
                post = self.trade(k, state, action)
                if not self.feasible(k, post):
                    raise ArithmeticError("В восстановленной стратегии нарушены ограничения.")
                plan.append((k, state, prob, histories, action, post))
                for j, p in enumerate(self.probs[k]):
                    child = self.transition(k, post, j)
                    old_p, old_count = following.get(child, (Fraction(0), 0))
                    following[child] = (old_p+prob*Fraction(p, self.P[k]), old_count+histories)
                    if len(following) > self.limits["max_states"]:
                        raise RuntimeError("Слишком много состояний при проверке стратегии.")
                self.check_time()
            current = following
        probability = sum(p for p, _ in current.values())
        expected = sum(prob*Fraction(sum(state), self.base*self.factor[-1])
                       for state, (prob, _) in current.items())
        if probability != 1 or expected != Fraction(self.result, self.value_scale[0]):
            raise ArithmeticError("Прямой проход не совпал с расчётом Беллмана.")
        return {"expected": expected, "probability": probability,
                "histories": sum(count for _, count in current.values()),
                "plan": plan}

def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

def save_tables(folder, solver, audit):
    values, actions = [], []
    for (k, state), value in sorted(solver.values.items()):
        state_label = solver.state_text(k, state)
        values.append((k+1, state_label, solver.value_text(k, value),
                       str(solver.policy[k, state]), "допустимо" if value is not None else "нет стратегии"))
        for row in solver.tables[k, state]:
            actions.append((k+1, state_label, str(row["action"]),
                            solver.value_text(k, row["upper"]), solver.value_text(k, row["value"]), row["status"]))
    write_csv(folder/"bellman_values.csv", ("Этап", "Состояние", "V", "Все оптимальные действия", "Статус"), values)
    write_csv(folder/"bellman_actions.csv", ("Этап", "Состояние", "Действие", "Верхняя оценка Q", "Точное Q", "Статус"), actions)
    plan = [(k+1, solver.state_text(k, state), text(p), count, str(u), solver.state_text(k, post))
            for k, state, p, count, u, post in audit["plan"]]
    write_csv(folder/"strategy.csv", ("Этап", "Состояние", "Вероятность", "Число историй", "Действие", "После сделок"), plan)

def main():
    folder = Path(__file__).with_name("results")
    table_files = ("bellman_values.csv", "bellman_actions.csv", "strategy.csv")
    try:
        folder.mkdir(exist_ok=True)
        for filename in table_files:
            write_csv(folder/filename, ("Статус",), [("Расчёт не завершён",)])
        solver = PortfolioDP(DATA, LIMITS, USE_BOUNDS)
        expected = solver.solve()
        if expected is None:
            save_tables(folder, solver, {"plan": []})
            print("Допустимой стратегии для всех сценариев положительной вероятности нет.")
            return 0
        check = solver.audit()
        save_tables(folder, solver, check)
        print("Критерий: максимальный ожидаемый итоговый доход.")
        print("Ожидаемый конечный капитал:", text(expected))
        print("Ожидаемый доход:", text(expected-solver.initial_total))
        print("\nОптимальная стратегия (первое из равнозначных действий):")
        for k, state, prob, _, action, post in check["plan"]:
            print(f"Этап {k+1}: {solver.state_text(k, state)}, вероятность {text(prob)}")
            print("  V:", solver.value_text(k, solver.values[k, state]), "; пакеты:", action)
            print("  после сделок:", solver.state_text(k, post))
        print("\nОграничения проверены. Сумма вероятностей:", text(check["probability"]))
        print("Сценарных историй:", check["histories"], "; прямой проход:", text(check["expected"]))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        error_types = ((ValueError, "invalid_data"), (RuntimeError, "runtime_error"),
                       (MemoryError, "limit_reached"),
                       (KeyboardInterrupt, "interrupted"), (OSError, "file_error"))
        name = next((code for kind, code in error_types if isinstance(error, kind)), "internal_error")
        message = f"{name}: {str(error) or type(error).__name__}"
        print("Расчёт не завершён:", message)
        try:
            for filename in table_files:
                write_csv(folder/filename, ("Ошибка расчёта",), [(message,)])
        except (OSError, MemoryError):
            print("Не удалось обновить таблицы. Старые файлы не относятся к этому запуску.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
