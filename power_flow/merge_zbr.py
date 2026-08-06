from typing import Dict, List, Optional, Set, Tuple, Union, Sequence
from collections import defaultdict
from matpow import MatpowerCase, BusEnergyBalance
import networkx as nx


def find_short_circuit_buses(
        case: MatpowerCase,
        id_to_bus: Dict[Union[int, str], BusEnergyBalance],
) -> Dict[int, int]:
    """Find short-circuited connected components across branches.

    Args:
        case: MATPOWER case object or dict.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.

    Returns:
        Tuple of (disconnected branch indices, list of connected component sets of bus IDs).
    """

    branch_data = case.branch
    g = nx.Graph()

    for idx, br in enumerate(branch_data):
        if not br.is_in_service:
            continue
        fb, tb = br.f_bus, br.t_bus
        if fb in id_to_bus and tb in id_to_bus:
            if br.br_r == 0 and br.br_x == 0:
                g.add_edge(fb, tb)
                br.br_status = 0

    components = [set(c) for c in nx.connected_components(g)]
    print(components)

    bus_del_to_keep = {}
    for island in components:
        bus_to_keep = list(island)[0]
        for bus in island:
            if bus != bus_to_keep:
                bus_del_to_keep[bus] = bus_to_keep

    return bus_del_to_keep


def open_branch_connect_to_del_buses(
        mpc: MatpowerCase, bus_del_to_keep: Dict[int, int]
) -> MatpowerCase:
    for br in mpc.branch:
        if br.f_bus in bus_del_to_keep:
            br.br_status = 0
            print(f"Open branch between {br.f_bus} and {br.t_bus} with x = {br.br_x}")

        if br.t_bus in bus_del_to_keep:
            br.br_status = 0
            print(f"Open branch between {br.f_bus} and {br.t_bus} with x = {br.br_x}")

    return mpc


def replace_bus_ids(mpc: MatpowerCase, bus_del_to_keep: Dict[int, int]) -> MatpowerCase:
    for br in mpc.branch:
        if br.f_bus in bus_del_to_keep:
            print(
                f"Replace branch f_bus from {br.f_bus} to {bus_del_to_keep[br.f_bus]}"
            )
            br.f_bus = bus_del_to_keep[br.f_bus]

        if br.t_bus in bus_del_to_keep:
            print(
                f"Replace branch t_bus from {br.t_bus} to {bus_del_to_keep[br.t_bus]}"
            )
            br.t_bus = bus_del_to_keep[br.t_bus]

    for gen in mpc.gen:
        if gen.gen_bus in bus_del_to_keep:
            print(
                f"Replace gen bus id {gen.gen_bus} with {bus_del_to_keep[gen.gen_bus]}"
            )
            gen.gen_bus = bus_del_to_keep[gen.gen_bus]

    bus_p = defaultdict(float)
    bus_q = defaultdict(float)
    bus_shunt = defaultdict(float)
    for bus in mpc.bus:
        if bus.bus_i in bus_del_to_keep:
            bus.bus_type = 4
            bus_p[bus_del_to_keep[bus.bus_i]] += bus.pd
            bus_q[bus_del_to_keep[bus.bus_i]] += bus.qd
            bus_shunt[bus_del_to_keep[bus.bus_i]] += bus.gs
            print(f"Set bus id {bus.bus_i} as isolated bus.")

    for bus in mpc.bus:
        if bus.bus_type == 4:
            continue
        bus.pd += bus_p.get(bus.bus_i, 0)
        bus.qd += bus_q.get(bus.bus_i, 0)
        bus.gs += bus_shunt.get(bus.bus_i, 0)

    return mpc


def open_branch_from_to_same_bus(mpc: MatpowerCase) -> MatpowerCase:
    for br in mpc.branch:
        if br.f_bus == br.t_bus:
            br.br_status = 0
            print(f"Open branch between {br.f_bus} and {br.t_bus} with x = {br.br_x}")
    return mpc


def merge_zero_impedance_branches(mpc: MatpowerCase, id_to_bus) -> MatpowerCase:
    bus_del_to_keep = find_short_circuit_buses(mpc, id_to_bus)
    mpc = open_branch_connect_to_del_buses(mpc, bus_del_to_keep)
    mpc = replace_bus_ids(mpc, bus_del_to_keep)
    mpc = open_branch_from_to_same_bus(mpc)

    return mpc
