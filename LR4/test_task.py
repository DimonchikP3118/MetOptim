import csv
import importlib.util
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portfolio", HERE / "task.py")
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)


def make_data(assets, cash, rates):
    n = len(assets)
    return {
        "names": tuple(f"Актив {i+1}" for i in range(n)),
        "initial": assets, "cash": cash,
        "minimum": (0,) * n, "lots": (1,) * n, "commissions": (0,) * n,
        "stages": ((("Рынок", 1, rates),),),
    }


def adaptive_example():
    return {
        "names": ("ЦБ1", "ЦБ2"), "initial": (2, 1), "cash": 1,
        "lots": (1, 1), "minimum": (1, 0), "commissions": ("0.1", "0.1"),
        "stages": (
            (("Рост", "0.6", (2, "1.1")),
             ("Спад", "0.4", ("0.5", "0.9"))),
            (("А", "0.5", ("0.6", "1.5")),
             ("Б", "0.5", ("0.4", "1.3"))),
        ),
    }


def brute_force(data):
    if len(data["stages"]) != 2:
        raise ValueError("Контрольный перебор рассчитан ровно на два этапа.")
    q = tuple(F(str(x)) for x in data["lots"])
    minimum = tuple(F(str(x)) for x in data["minimum"])
    fee = tuple(a * F(str(b)) for a, b in zip(q, data["commissions"]))
    initial = tuple(F(str(x)) for x in data["initial"]) + (F(str(data["cash"])),)
    stages = [tuple((F(str(p)), tuple(F(str(r)) for r in rates))
                    for _, p, rates in rows if F(str(p)) > 0)
              for rows in data["stages"]]

    def actions(state):
        ranges = []
        for a, lot, m in zip(state[:-1], q, minimum):
            lower = (m - a) / lot
            low = -((-lower.numerator) // lower.denominator)
            high = (sum(state) - a) // lot
            ranges.append(range(low, high + 1))
        result = []
        for u in product(*ranges):
            assets = tuple(a + lot * v for a, lot, v in zip(state[:-1], q, u))
            cash = state[-1] - sum(lot*v + f*abs(v) for lot, f, v in zip(q, fee, u))
            if cash >= 0 and all(a >= m for a, m in zip(assets, minimum)):
                result.append((u, assets + (cash,)))
        return result

    def market(post, rates):
        return tuple(a*r for a, r in zip(post[:-1], rates)) + (post[-1],)

    best, best_fixed = None, None
    policies, count = set(), 0
    for first, post in actions(initial):
        branches = tuple((p, market(post, r)) for p, r in stages[0])
        observed = tuple(sorted({s for _, s in branches}))
        menus = [actions(s) for s in observed]
        for choices in product(*menus):
            count += 1
            selected = dict(zip(observed, choices))
            value = F(0)
            for p1, state in branches:
                _, post2 = selected[state]
                for p2, rates in stages[1]:
                    value += p1 * p2 * sum(market(post2, rates))
            signature = (first, tuple((s, selected[s][0]) for s in observed))
            if best is None or value > best:
                best, policies = value, {signature}
            elif value == best:
                policies.add(signature)
            if len({u for u, _ in choices}) == 1:
                if best_fixed is None or value > best_fixed:
                    best_fixed = value
    return {"value": best, "policies": policies, "strategies": count,
            "fixed_plan_value": best_fixed}



class TestRequiredChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = []

    def record(self, label, data, expected, actual):
        passed = expected == actual
        self.records.append({"Проверка": label, "Входные данные": data,
                             "Ожидаемый результат": expected, "Фактический результат": actual,
                             "Статус": "PASS" if passed else "FAIL"})
        self.assertEqual(expected, actual, label)

    @classmethod
    def tearDownClass(cls):
        folder = HERE / "test_results_required"
        folder.mkdir(exist_ok=True)
        with (folder / "checks.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(("Проверка", "Входные данные", "Ожидаемый результат", "Фактический результат", "Статус"))
            for row in cls.records:
                writer.writerow([row["Проверка"]] + [json.dumps(row[key], ensure_ascii=False, default=str)
                    for key in ("Входные данные", "Ожидаемый результат", "Фактический результат")] + [row["Статус"]])
        (folder / "checks.json").write_text(json.dumps(cls.records, ensure_ascii=False,
                                                       default=str, indent=2), encoding="utf-8")

    def test_01_independent_brute_force(self):
        data = adaptive_example()
        check = brute_force(data)
        first_actions = tuple(sorted({p[0] for p in check["policies"]}))
        for bounds in (True, False):
            with self.subTest(use_bounds=bounds):
                solver = program.PortfolioDP(data, use_bounds=bounds)
                value = solver.solve()
                audit = solver.audit()
                second = tuple(sorted(
                    (tuple(F(x, solver.base * solver.factor[k]) for x in state), u)
                    for k, state, _, _, u, _ in audit["plan"] if k == 1))
                policy = (solver.policy[0, solver.initial][0], second)
                actual = {"перебор": check["value"], "ДП": value,
                          "число стратегий": check["strategies"],
                          "фиксированный план": check["fixed_plan_value"],
                          "стратегия найдена перебором": policy in check["policies"],
                          "все первые действия совпали": solver.policy[0, solver.initial] == first_actions,
                          "адаптивность даёт выигрыш": value > check["fixed_plan_value"],
                          "прямой проход": audit["expected"]}
                expected = {"перебор": F(636, 125), "ДП": F(636, 125),
                            "число стратегий": 130, "фиксированный план": F(957, 250),
                            "стратегия найдена перебором": True,
                            "все первые действия совпали": True,
                            "адаптивность даёт выигрыш": True, "прямой проход": F(636, 125)}
                self.record("8.2. Адаптивный перебор; отсечения=" + str(bounds),
                            data, expected, actual)

    def test_02_zero_resource(self):
        data = make_data((0,), 0, (1,))
        solver = program.PortfolioDP(data)
        self.record("8.3.1. Нулевой капитал", data, (F(0), ((0,),)),
                    (solver.solve(), solver.policy[0, solver.initial]))

    def test_03_one_asset(self):
        data = make_data((4,), 2, (2,))
        solver = program.PortfolioDP(data)
        self.record("8.3.2. Один актив", data, (F(12), ((2,),)),
                    (solver.solve(), solver.policy[0, solver.initial]))

    def test_04_zero_income(self):
        data = make_data((2, 3), 1, (1, 1))
        solver = program.PortfolioDP(data)
        value = solver.solve()
        self.record("8.3.3. Нулевая доходность", data, (F(6), F(0)),
                    (value, value-solver.initial_total))

    def test_05_several_optimal_plans(self):
        data = make_data((2,), 1, (1,))
        solver = program.PortfolioDP(data)
        self.record("8.3.4. Несколько оптимальных планов", data,
                    (F(3), ((-2,), (-1,), (0,), (1,))),
                    (solver.solve(), solver.policy[0, solver.initial]))

    def test_06_all_money_in_one_asset(self):
        data = make_data((4, 4), 0, (2, "0.5"))
        solver = program.PortfolioDP(data)
        self.record("8.3.5. Все средства в один актив", data, (F(16), ((4, -4),)),
                    (solver.solve(), solver.policy[0, solver.initial]))

    def test_07_invalid_input(self):
        cases = [("Отрицательные средства", make_data((4,), -1, (1,))),
                 ("Неполные коэффициенты", make_data((4, 4), 0, (1,)))]
        for label, data in cases:
            with self.subTest(case=label):
                try:
                    program.PortfolioDP(data)
                except ValueError:
                    actual = "ValueError"
                else:
                    actual = "Ошибка не обнаружена"
                self.record("8.3.6. " + label, data, "ValueError", actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
