import csv
import gc
import json
import platform
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

from task import DATA, PortfolioDP, save_tables


def table_memory(*tables):
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple, set, frozenset)):
            total += sum(size(x) for x in value)
        return total

    return sum(size(t) for t in tables)


def main():
    folder = Path(__file__).with_name("results")
    folder.mkdir(exist_ok=True)
    start = perf_counter()
    solver = PortfolioDP(DATA)
    value = solver.solve()
    solve_seconds = perf_counter() - start
    start = perf_counter()
    audit = solver.audit()
    audit_seconds = perf_counter() - start
    memory_bytes = table_memory(solver.values, solver.policy, solver.tables)
    start = perf_counter()
    save_tables(folder, solver, audit)
    export_seconds = perf_counter() - start
    files = {}
    for name in ("bellman_values.csv", "bellman_actions.csv", "strategy.csv"):
        path = folder / name
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = sum(1 for _ in csv.reader(f, delimiter=";")) - 1
        files[name] = {"rows_without_header": rows, "bytes": path.stat().st_size}
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "expected_wealth": str(value),
        "expected_income": str(value - solver.initial_total),
        "solve_seconds": solve_seconds, "audit_seconds": audit_seconds,
        "export_seconds": export_seconds, **solver.stats,
        "feasible_actions": sum(len(rows) for rows in solver.tables.values()),
        "evaluated_actions": sum(row["status"] == "вычислено"
                                 for rows in solver.tables.values() for row in rows),
        "pruned_actions": sum(row["status"] == "отсечено по верхней оценке"
                              for rows in solver.tables.values() for row in rows),
        "table_objects_bytes": memory_bytes,
        "strategy_states": len(audit["plan"]), "histories": audit["histories"],
        "files": files, "csv_total_bytes": sum(v["bytes"] for v in files.values()),
    }
    del solver, audit
    gc.collect()
    tracemalloc.start()
    memory_solver = PortfolioDP(DATA)
    memory_value = memory_solver.solve()
    memory_solver.audit()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if memory_value != value:
        raise ArithmeticError("Результаты двух запусков не совпали.")
    result["python_peak_bytes"] = peak
    (folder / "measurements.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
