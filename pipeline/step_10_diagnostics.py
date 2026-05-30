"""
10_diagnostics: всі RL-діагностичні перевірки в одному кроці.

Об'єднує колишні кроки 10b–10f:
  run_schedule_debug()   — якість розкладів локального маршруту (10b)
  run_recompute_check()  — порівняння baseline I*_peak з RL recompute (10c)
  run_routes_freq()      — зведення частот по всіх маршрутах (10d)
  run_target_group()     — аналіз target-групи перед навчанням (10e)
  run_action_probe()     — brute-force one-step probe (10f)

run() виконує всі п'ять по порядку. Кожну можна викликати окремо.
"""


def run_schedule_debug() -> None:
    """Якість розкладів локальної підмережі для target-закладу (10b)."""
    import pipeline.step_10b_h327_route_debug as _m
    _m.run()


def run_recompute_check() -> None:
    """Порівнює baseline I*_peak з RL recompute при базових частотах (10c)."""
    import pipeline.step_10c_rl_recompute_check as _m
    _m.run()


def run_routes_freq() -> None:
    """Виводить поточні частоти всіх маршрутів будніх днів (10d)."""
    import pipeline.step_10d_all_routes_freq_debug as _m
    _m.run()


def run_target_group() -> None:
    """Аналізує target-групу: маршрути, заклади, простір для переносу (10e)."""
    import pipeline.step_10e_target_group_debug as _m
    _m.run()


def run_action_probe() -> None:
    """Brute-force перебір всіх одно-крокових donor→receiver дій (10f)."""
    import pipeline.step_10f_rl_action_probe as _m
    _m.run()


def run() -> None:
    print("=== 10_diagnostics: schedule debug ===")
    try:
        run_schedule_debug()
    except Exception as exc:
        print(f"  [WARN] schedule_debug пропущено: {exc}")

    print("\n=== 10_diagnostics: recompute check ===")
    try:
        run_recompute_check()
    except Exception as exc:
        print(f"  [WARN] recompute_check пропущено: {exc}")

    print("\n=== 10_diagnostics: routes freq ===")
    try:
        run_routes_freq()
    except Exception as exc:
        print(f"  [WARN] routes_freq пропущено: {exc}")

    print("\n=== 10_diagnostics: target group ===")
    try:
        run_target_group()
    except Exception as exc:
        print(f"  [WARN] target_group пропущено: {exc}")

    print("\n=== 10_diagnostics: action probe ===")
    try:
        run_action_probe()
    except Exception as exc:
        print(f"  [WARN] action_probe пропущено: {exc}")


if __name__ == "__main__":
    run()
