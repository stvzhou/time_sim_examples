from typing import List
from collections import defaultdict
from matpow import MatpowerCase
from calc_rx import calc_flow_from_bus, calc_flow_to_bus

PJM_AREAS = [
    "AE",
    "AEP",
    "APS",
    "BGE",
    "CE",
    "DAY",
    "DEO&K",
    "DLCO",
    "DP&L",
    "DVP",
    "EKPC",
    "FE",
    "HE",
    "JCPL",
    "ME",
    "OVEC",
    "PECO",
    "PENELEC",
    "PEPCO",
    "PJM-MTX",
    "PL",
    "PSEG",
    "RECO",
    "SMECO",
    "UGI",
]


def extract_areas(mpc: MatpowerCase, areas: List[str]):
    print("\n" + "=" * 140)
    assert all(area in mpc.area_name_to_id for area in areas)
    area_ids = {mpc.area_name_to_id[area] for area in areas}
    buses = [b for b in mpc.bus if b.bus_area in area_ids]
    bus_ids = {b.bus_i for b in buses}
    gens = [g for g in mpc.gen if g.gen_bus in bus_ids]
    branches = []
    dummy_p = defaultdict(float)
    dummy_q = defaultdict(float)
    id_to_bus = {b.bus_i: b for b in mpc.bus}
    for br in mpc.branch:
        vsm = id_to_bus[br.f_bus].vm
        vrm = id_to_bus[br.t_bus].vm
        vsa = id_to_bus[br.f_bus].va
        vra = id_to_bus[br.t_bus].va

        if br.f_bus in bus_ids and br.t_bus in bus_ids:
            branches.append(br)
        elif br.f_bus in bus_ids:
            flow = calc_flow_from_bus(vsm=vsm, vrm=vrm, vsa=vsa, vra=vra, r=br.br_r, x=br.br_x, shift=br.shift,
                                      b_total=br.br_b, ratio=br.tap, )
            dummy_p[br.f_bus] += flow.p
            dummy_q[br.f_bus] += flow.q
        elif br.t_bus in bus_ids:
            flow = calc_flow_to_bus(vsm=vsm, vrm=vrm, vsa=vsa, vra=vra, r=br.br_r, x=br.br_x, shift=br.shift,
                                    b_total=br.br_b, ratio=br.tap)
            dummy_p[br.t_bus] += flow.p
            dummy_q[br.t_bus] += flow.q

    for bus in buses:
        bus.pd += dummy_p.get(bus.bus_i, 0)
        bus.qd += dummy_q.get(bus.bus_i, 0)

    if len([b for b in buses if b.bus_type == 3]) == 0:
        max_gen_bus = list(sorted(gens, key=lambda g: g.pmax, reverse=True))[0]
        for bus in buses:
            if bus.bus_i == max_gen_bus.gen_bus:
                bus.bus_type = 3
                print("Set bus {} to slack bus with pmax = {}".format(bus.bus_i, max_gen_bus.pmax))

    new_mpc = MatpowerCase(
        version="2",
        baseMVA=mpc.baseMVA,
        bus=buses,
        gen=gens,
        branch=branches,
        bus_miss={k: v for k, v in mpc.bus_miss.items() if k in bus_ids},
    )

    print(f"Extracted {len(buses)} buses from {len(mpc.bus)} for {len(areas)} areas")
    print("\n" + "=" * 140)

    return new_mpc.extract_main_island()
