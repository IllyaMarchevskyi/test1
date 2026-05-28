"""Допоміжні функції для матриці сумісності RL-перерозподілу."""

from __future__ import annotations


DEFAULT_TRANSPORT_TYPES = ("bus", "trol", "tram", "metro")


def parse_config_list(value) -> list[str]:
    """Повертає список рядків; підтримує і ["trol, bus"], і ["trol", "bus"]."""
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []

    result: list[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            normalized = part.strip().lower()
            if normalized:
                result.append(normalized)
    return result


def load_transfer_compatibility(
    rl_cfg: dict,
    transport_types: tuple[str, ...] = DEFAULT_TRANSPORT_TYPES,
) -> dict[str, set[str]]:
    """
    Завантажує donor -> allowed receivers.

    Якщо [rl.transfer_compatibility] є в config.toml, він має пріоритет.
    Інакше лишається стара поведінка:
    - allow_cross_type_transfers=true: кожен тип може перейти в кожен;
    - false: тільки в межах одного типу.
    """
    raw_matrix = rl_cfg.get("transfer_compatibility")
    if isinstance(raw_matrix, dict) and raw_matrix:
        matrix: dict[str, set[str]] = {}
        for donor, receivers in raw_matrix.items():
            donor_key = str(donor).strip().lower()
            allowed = set(parse_config_list(receivers))
            if donor_key and allowed:
                matrix[donor_key] = allowed
        return matrix

    all_types = {str(tt).strip().lower() for tt in transport_types if str(tt).strip()}
    allow_cross_type = bool(rl_cfg.get("allow_cross_type_transfers", False))
    if allow_cross_type:
        return {tt: set(all_types) for tt in all_types}
    return {tt: {tt} for tt in all_types}


def transfer_compatibility_for_run(rl_cfg: dict) -> dict[str, list[str]]:
    """JSON-friendly представлення матриці для run_config/result-файлів."""
    matrix = load_transfer_compatibility(rl_cfg)
    return {donor: sorted(receivers) for donor, receivers in sorted(matrix.items())}


def is_transfer_allowed(
    donor_transport: str,
    receiver_transport: str,
    compatibility: dict[str, set[str]],
) -> bool:
    donor = str(donor_transport).strip().lower()
    receiver = str(receiver_transport).strip().lower()
    allowed = compatibility.get(donor)
    if allowed is None:
        # Без явного правила тип лишається сумісним тільки сам із собою.
        return donor == receiver
    return receiver in allowed


def build_transfer_actions(
    route_transports: list[str],
    compatibility: dict[str, set[str]],
) -> list[tuple[int, int]]:
    """Будує всі валідні donor->receiver пари за матрицею сумісності."""
    actions: list[tuple[int, int]] = []
    for donor_idx, donor_transport in enumerate(route_transports):
        for receiver_idx, receiver_transport in enumerate(route_transports):
            if donor_idx == receiver_idx:
                continue
            if is_transfer_allowed(donor_transport, receiver_transport, compatibility):
                actions.append((donor_idx, receiver_idx))
    return actions


def count_transfer_actions_for_routes(
    route_ids: set[str],
    route_transport_by_id: dict[str, str],
    compatibility: dict[str, set[str]],
) -> int:
    """Рахує кількість можливих donor->receiver дій для набору route_id."""
    route_list = sorted(str(route_id) for route_id in route_ids)
    route_transports = [route_transport_by_id.get(route_id, "unknown") for route_id in route_list]
    return len(build_transfer_actions(route_transports, compatibility))
