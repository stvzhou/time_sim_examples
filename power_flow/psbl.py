import pypowsybl.network as pn
import pypowsybl.loadflow as lf
import pypowsybl.report as report
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List
import networkx as nx


@dataclass
class BusInfo:
    id: str
    substation_id: str
    v_mag: float


def get_bus_substation(network: pn, bus_id: str):
    buses = network.get_buses()
    volt_levels = network.get_voltage_levels()
    bus = buses.loc[bus_id]["voltage_level_id"]

    return volt_levels.loc[bus]["substation_id"]


def subs_with_volt_violations(
    network: pn,
    top_n: int,
    connected_component: int = 0,
    synchronous_component: int = 0,
    violate_upper_limit: bool = True,
):
    buses = network.get_buses()
    buses = buses[
        (buses["connected_component"] == connected_component)
        & (buses["synchronous_component"] == synchronous_component)
    ]
    buses = buses.sort_values(by="v_mag", ascending=(not violate_upper_limit)).head(
        top_n
    )
    volt_levels = network.get_voltage_levels()
    result = []
    for idx, row in buses.iterrows():
        result.append(
            BusInfo(
                id=idx,
                substation_id=volt_levels.loc[row.voltage_level_id]["substation_id"],
                v_mag=row.v_mag,
            )
        )

    return result


@dataclass
class Flow:
    p: float
    q: float


def calc_flow_into_line(vsm, vrm, vsa, vra, r, x, b, base_mva=100.0) -> Flow:
    z_square = r**2 + x**2
    delta = np.deg2rad(vsa) - np.deg2rad(vra)
    ps = vsm / z_square * (vsm * r - r * vrm * np.cos(delta) + x * vrm * np.sin(delta))
    qs = (
        vsm
        / z_square
        * (
            vsm * x
            - x * vrm * np.cos(delta)
            + r * vrm * np.sin(delta)
            - vsm * z_square * b / 2
        )
    )

    return Flow(ps * base_mva, qs * base_mva)


def calc_flow_from_line(vsm, vrm, vsa, vra, r, x, b, base_mva=100.0):
    z_square = r**2 + x**2
    delta = np.deg2rad(vsa) - np.deg2rad(vra)
    pr = vrm / z_square * (-vrm * r + r * vsm * np.cos(delta) + x * vsm * np.sin(delta))
    qr = (
        vrm
        / z_square
        * (
            -vrm * x
            + x * vsm * np.cos(delta)
            - r * vsm * np.sin(delta)
            + vrm * z_square * b / 2
        )
    )

    return Flow(pr * base_mva, qr * base_mva)


@dataclass
class BusEnergyBalance:
    vm: float
    va: float
    connected_component: int
    synchronous_component: int
    outgoing_flows: Dict[str, Flow] = field(default_factory=dict)
    incoming_flows: Dict[str, Flow] = field(default_factory=dict)
    shunts: Dict[str, Flow] = field(default_factory=dict)
    facts: Dict[str, Flow] = field(default_factory=dict)
    loads: Dict[str, Flow] = field(default_factory=dict)
    generations: Dict[str, Flow] = field(default_factory=dict)
    hvdc: Dict[str, Flow] = field(default_factory=dict)


def calc_bus_balance(bus: BusEnergyBalance) -> Flow:
    p, q = 0, 0
    for flow in bus.outgoing_flows.values():
        p += flow.p
        q += flow.q
    for flow in bus.incoming_flows.values():
        p -= flow.p
        q -= flow.q
    for flow in bus.shunts.values():
        p -= flow.p
        q -= flow.q
    for flow in bus.facts.values():
        p -= flow.p
        q -= flow.q
    for flow in bus.loads.values():
        p += flow.p
        q += flow.q
    for flow in bus.generations.values():
        p -= flow.p
        q -= flow.q
    for flow in bus.hvdc.values():
        p += flow.p
        q += flow.q

    return Flow(p, q)


def get_bus_generations(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_generators().iterrows():
        if row["bus_id"]:
            id_to_bus[row["bus_id"]].generations[idx] = Flow(
                row["target_p"], row["target_q"]
            )

    return id_to_bus


def get_bus_loads(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_loads().iterrows():
        if row["bus_id"]:
            id_to_bus[row["bus_id"]].loads[idx] = Flow(row["p0"], row["q0"])

    return id_to_bus


def get_facts(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_static_var_compensators().iterrows():
        if row["bus_id"]:
            id_to_bus[row["bus_id"]].facts[idx] = Flow(0, row["target_q"])

    return id_to_bus


def get_shunts(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_shunt_compensators().iterrows():
        if row["bus_id"]:
            id_to_bus[row["bus_id"]].shunts[idx] = Flow(0, row["b"])

    return id_to_bus


def get_line_flows(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_lines().iterrows():
        if not row["connected1"] or not row["connected2"]:
            continue
        bus1 = row["bus1_id"]
        bus2 = row["bus2_id"]
        vsm = id_to_bus[bus1].vm
        vsa = id_to_bus[bus1].va
        vrm = id_to_bus[bus2].vm
        vra = id_to_bus[bus2].va
        r = row["r"]
        x = row["x"]
        b = row["b1"] + row["b2"]
        id_to_bus[bus1].outgoing_flows[idx] = calc_flow_into_line(
            vsm, vrm, vsa, vra, r, x, b
        )
        id_to_bus[bus2].incoming_flows[idx] = calc_flow_from_line(
            vsm, vrm, vsa, vra, r, x, b
        )

    return id_to_bus


def get_three_winds_trans(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_3_windings_transformers().iterrows():
        if sum([row["connected1"], row["connected2"], row["connected3"]]) < 2:
            continue

        if row["connected1"]:
            bus1 = row["bus1_id"]
            bus2 = idx
            vsm = id_to_bus[bus1].vm
            vsa = id_to_bus[bus1].va
            vrm = id_to_bus[bus2].vm
            vra = id_to_bus[bus2].va
            r = row["r1"]
            x = row["x1"]
            b = row["b1"]
            id_to_bus[bus1].outgoing_flows[idx] = calc_flow_into_line(
                vsm, vrm, vsa, vra, r, x, b
            )
            id_to_bus[bus2].incoming_flows[idx] = calc_flow_from_line(
                vsm, vrm, vsa, vra, r, x, b
            )

        if row["connected2"]:
            bus1 = row["bus2_id"]
            bus2 = idx
            vsm = id_to_bus[bus1].vm
            vsa = id_to_bus[bus1].va
            vrm = id_to_bus[bus2].vm
            vra = id_to_bus[bus2].va
            r = row["r2"]
            x = row["x2"]
            b = row["b2"]
            id_to_bus[bus1].outgoing_flows[idx] = calc_flow_into_line(
                vsm, vrm, vsa, vra, r, x, b
            )
            id_to_bus[bus2].incoming_flows[idx] = calc_flow_from_line(
                vsm, vrm, vsa, vra, r, x, b
            )

        if row["connected3"]:
            bus1 = row["bus3_id"]
            bus2 = idx
            vsm = id_to_bus[bus1].vm
            vsa = id_to_bus[bus1].va
            vrm = id_to_bus[bus2].vm
            vra = id_to_bus[bus2].va
            r = row["r3"]
            x = row["x3"]
            b = row["b3"]
            id_to_bus[bus1].outgoing_flows[idx] = calc_flow_into_line(
                vsm, vrm, vsa, vra, r, x, b
            )
            id_to_bus[bus2].incoming_flows[idx] = calc_flow_from_line(
                vsm, vrm, vsa, vra, r, x, b
            )

    return id_to_bus


def get_two_winds_trans(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    for idx, row in net.get_2_windings_transformers().iterrows():
        if not row["connected1"] or not row["connected2"]:
            continue
        bus1 = row["bus1_id"]
        bus2 = row["bus2_id"]
        vsm = id_to_bus[bus1].vm
        vsa = id_to_bus[bus1].va
        vrm = id_to_bus[bus2].vm
        vra = id_to_bus[bus2].va
        r = row["r"]
        x = row["x"]
        b = row["b"]
        id_to_bus[bus1].outgoing_flows[idx] = calc_flow_into_line(
            vsm, vrm, vsa, vra, r, x, b
        )
        id_to_bus[bus2].incoming_flows[idx] = calc_flow_from_line(
            vsm, vrm, vsa, vra, r, x, b
        )

    return id_to_bus


def get_star_bus_from_raw(file_path: str) -> Dict[str, BusEnergyBalance]:
    with open(file_path, "r") as f:
        lines = f.readlines()
    star_buses = {}
    i = 0
    while "BEGIN TRANSFORMER DATA" not in lines[i]:
        i += 1

    i += 1
    while "END OF TRANSFORMER DATA, BEGIN AREA DATA" not in lines[i]:
        buses = lines[i].split(",")
        bus_i = int(buses[0].strip())
        bus_j = int(buses[1].strip())
        bus_k = int(buses[2].strip())
        ckt = buses[3].replace("'", "")

        if bus_k == 0:
            i += 4
        else:
            bus_name = f"T-{bus_i}-{bus_j}-{bus_k}-{ckt}"
            volts = lines[i + 1].split(",")
            star_buses[bus_name] = BusEnergyBalance(
                vm=float(volts[-2].strip()),
                va=float(volts[-1].strip()),
                connected_component=-1,
                synchronous_component=-1,
            )
            i += 5

    return star_buses


def get_hvdc_flow(
    net: pn, id_to_bus: Dict[str, BusEnergyBalance]
) -> Dict[str, BusEnergyBalance]:
    station_to_bus = {}
    station_to_power_factor = {}
    for idx, row in net.get_lcc_converter_stations().iterrows():
        if row["bus_id"]:
            station_to_bus[idx] = row["bus_id"]
            station_to_power_factor[idx] = row["power_factor"]

    for idx, row in net.get_vsc_converter_stations().iterrows():
        if row["bus_id"]:
            station_to_bus[idx] = row["bus_id"]
            station_to_power_factor[idx] = 0.0

    for idx, row in net.get_hvdc_lines().iterrows():
        if not row["connected1"] or not row["connected2"]:
            continue
        bus1 = station_to_bus[row["converter_station1_id"]]
        bus2 = station_to_bus[row["converter_station2_id"]]
        s1 = (
            row["target_p"] / station_to_power_factor[row["converter_station1_id"]]
            if station_to_power_factor[row["converter_station1_id"]] != 0
            else row["target_p"]
        )
        q1 = np.sqrt(s1**2 - row["target_p"] ** 2)
        s2 = (
            row["target_p"] / station_to_power_factor[row["converter_station2_id"]]
            if station_to_power_factor[row["converter_station2_id"]] != 0
            else row["target_p"]
        )
        q2 = np.sqrt(s2**2 - row["target_p"] ** 2)
        id_to_bus[bus1].hvdc[idx] = Flow(row["target_p"], q1)
        id_to_bus[bus2].hvdc[idx] = Flow(-row["target_p"], -q2)

    return id_to_bus


def open_low_impeadance_branches(net: pn, id_to_bus: Dict[str, BusEnergyBalance]) -> pn:
    for idx, row in net.get_2_windings_transformers().iterrows():
        if row["connected1"] and row["connected2"] and abs(row["x"]) < 0.0002:
            bus1 = id_to_bus[row["bus1_id"]]
            bus2 = id_to_bus[row["bus2_id"]]
            if (
                abs(np.radians(bus1.va) - np.radians(bus2.va)) / abs(row["x"]) > 30
                and abs(bus1.va - bus2.va) > 0.01
            ):
                print("Open low impeadance transformer: " + idx)
                net.disconnect(idx)

    for idx, row in net.get_lines().iterrows():
        if row["connected1"] and row["connected2"] and abs(row["x"]) < 0.0002:
            bus1 = id_to_bus[row["bus1_id"]]
            bus2 = id_to_bus[row["bus2_id"]]
            if (
                abs(np.radians(bus1.va) - np.radians(bus2.va)) / abs(row["x"]) > 30
                and abs(bus1.va - bus2.va) > 0.01
            ):
                print("Open low impeadance lines: " + idx)
                net.disconnect(idx)

    return net


def is_buses_short_circuited(bus1: BusEnergyBalance, bus2: BusEnergyBalance):
    if abs(bus1.vm - bus2.vm) < 0.01 and abs(bus1.va - bus2.va) < 0.01:
        return True

    return False


def find_short_circuit_buses(net: pn, id_to_bus: Dict[str, BusEnergyBalance]) -> pn:
    G = nx.Graph()
    for idx, row in net.get_lines().iterrows():
        if not row["connected1"] or not row["connected2"]:
            continue
        bus1 = id_to_bus[row["bus1_id"]]
        bus2 = id_to_bus[row["bus2_id"]]
        if is_buses_short_circuited(bus1, bus2):
            G.add_edge(row["bus1_id"], row["bus2_id"])
            net.disconnect(idx)

    for idx, row in net.get_2_windings_transformers().iterrows():
        if not row["connected1"] or not row["connected2"]:
            continue
        bus1 = id_to_bus[row["bus1_id"]]
        bus2 = id_to_bus[row["bus2_id"]]
        if is_buses_short_circuited(bus1, bus2):
            G.add_edge(row["bus1_id"], row["bus2_id"])
            net.disconnect(idx)

    for idx, row in net.get_3_windings_transformers().iterrows():
        star = id_to_bus[idx]

        if row["bus1_id"]:
            bus1 = id_to_bus[row["bus1_id"]]
            if is_buses_short_circuited(bus1, star):
                G.add_edge(row["bus1_id"], idx)
                net.update_3_windings_transformers(id=idx, connected1=False)

        if row["bus2_id"]:
            bus2 = id_to_bus[row["bus2_id"]]
            if is_buses_short_circuited(bus2, star):
                G.add_edge(row["bus2_id"], idx)
                net.update_3_windings_transformers(id=idx, connected2=False)

        if row["bus3_id"]:
            bus3 = id_to_bus[row["bus3_id"]]
            if is_buses_short_circuited(bus3, star):
                G.add_edge(row["bus3_id"], idx)
                net.update_3_windings_transformers(id=idx, connected3=False)

    return net, [set(c) for c in nx.connected_components(G)]


def get_equivalent_buses(islands, star_buses) -> Dict[str, str]:
    result = {}
    for island in islands:
        bus_to_keep = list(island)[0]
        for bus in island:
            if bus in star_buses:
                bus_to_keep = bus
                break

        for bus in island:
            if bus != bus_to_keep:
                result[bus] = bus_to_keep

    return result


if __name__ == "__main__":
    raw_file = (
        "/usr/local/google/home/sxzhou/Downloads/2025 Series RTEP 2030 SUM_06182025.raw"
    )
    id_to_bus = get_star_bus_from_raw(raw_file)
    star_buses = set(id_to_bus.keys())
    network = pn.load(
        raw_file,
        {"psse.import.ignore-base-voltage": "true"},
    )

    for idx, row in network.get_buses().iterrows():
        id_to_bus[idx] = BusEnergyBalance(
            vm=row["v_mag"],
            va=row["v_angle"],
            connected_component=row["connected_component"],
            synchronous_component=row["synchronous_component"],
        )

    network = open_low_impeadance_branches(network, id_to_bus)
    net, short_circuit_buses = find_short_circuit_buses(network, id_to_bus)
    bus_mappings = get_equivalent_buses(short_circuit_buses, star_buses)

    # rp = report.ReportNode()
    # hvdcs = network.get_hvdc_lines().index.tolist()
    # pn.remove_hvdc_lines(network, hvdcs)
    # results = lf.run_ac(
    #     network,
    #     report_node=rp,
    #     parameters=lf.Parameters(
    #         component_mode=lf.ComponentMode.MAIN_SYNCHRONOUS,
    #         voltage_init_mode=lf.VoltageInitMode.PREVIOUS_VALUES,
    #         shunt_compensator_voltage_control_on=False,
    #         transformer_voltage_control_on=False,
    #     ),
    # )
    # print(rp)

    id_to_bus = get_bus_generations(network, id_to_bus)
    id_to_bus = get_bus_loads(network, id_to_bus)
    id_to_bus = get_facts(network, id_to_bus)
    id_to_bus = get_shunts(network, id_to_bus)
    id_to_bus = get_line_flows(network, id_to_bus)
    id_to_bus = get_two_winds_trans(network, id_to_bus)
    id_to_bus = get_three_winds_trans(network, id_to_bus)
    id_to_bus = get_hvdc_flow(network, id_to_bus)

    result = sorted(id_to_bus.items(), key=lambda x: calc_bus_balance(x[1]).p)
    pass
