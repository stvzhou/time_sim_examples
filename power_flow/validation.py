"""Bus energy balance validation for MATPOWER data structures.

Follows the energy balance validation pattern in psbl.py adapted for MATPOWER 5.0 case format:
- Flow dataclass for (P, Q) active and reactive power
- BusEnergyBalance dataclass for per-bus tracking of all connected injections/withdrawals
- calc_flow_into_branch / calc_flow_from_branch for transmission lines and transformers (with tap and phase shift)
- calc_bus_balance for net nodal (P, Q) mismatch evaluation
- Data extraction from MATPOWER data structures (MatpowerCase, dict, numpy matrices)
- Low impedance branch detection, short-circuit bus identification, and topology aggregation
- Validation routines, summary reports, and violation detection
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import networkx as nx

# Ensure power_flow directory is in python path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from matpow import (
        BUS_AREA,
        BUS_I,
        BUS_TYPE,
        BASE_KV,
        BS,
        F_BUS,
        GEN_BUS,
        GEN_STATUS,
        GS,
        PD,
        PG,
        QD,
        QG,
        SHIFT,
        T_BUS,
        TAP,
        VA,
        VM,
        ZONE,
        Branch,
        Bus,
        Generator,
        MatpowerCase,
    )
except ImportError:
    # Fallback column index constants (MATPOWER 5.0)
    BUS_I = 0
    BUS_TYPE = 1
    PD = 2
    QD = 3
    GS = 4
    BS = 5
    BUS_AREA = 6
    VM = 7
    VA = 8
    BASE_KV = 9
    ZONE = 10

    GEN_BUS = 0
    PG = 1
    QG = 2
    GEN_STATUS = 7

    F_BUS = 0
    T_BUS = 1
    TAP = 8
    SHIFT = 9


# =============================================================================
# Core Dataclasses
# =============================================================================


@dataclass
class Flow:
    """Active (P) and Reactive (Q) power flow representation.

    Attributes:
        p: Active power (MW)
        q: Reactive power (MVAr)
    """

    p: float = 0.0
    q: float = 0.0

    @property
    def s(self) -> float:
        """Apparent power magnitude (MVA) S = sqrt(P^2 + Q^2)."""
        return math.sqrt(self.p**2 + self.q**2)

    def __add__(self, other: Flow) -> Flow:
        return Flow(self.p + other.p, self.q + other.q)

    def __sub__(self, other: Flow) -> Flow:
        return Flow(self.p - other.p, self.q - other.q)

    def __neg__(self) -> Flow:
        return Flow(-self.p, -self.q)

    def is_zero(self, tol: float = 1e-8) -> bool:
        """Return True if both P and Q are within tolerance of zero."""
        return abs(self.p) <= tol and abs(self.q) <= tol

    def to_tuple(self) -> Tuple[float, float]:
        """Return (p, q) tuple."""
        return (self.p, self.q)

    def to_dict(self) -> Dict[str, float]:
        """Return dictionary representation."""
        return {"p": self.p, "q": self.q, "s": self.s}


@dataclass
class BusInfo:
    """Summary of bus voltage, area, and energy balance metrics.

    Attributes:
        id: Bus identifier (integer or string)
        bus_type: MATPOWER bus type (1=PQ, 2=PV, 3=Slack/Ref, 4=Isolated)
        v_mag: Voltage magnitude (p.u.)
        v_angle: Voltage angle (degrees)
        p_mismatch: Active power mismatch (MW)
        q_mismatch: Reactive power mismatch (MVAr)
        base_kv: Base voltage (kV)
        bus_area: Area number
        zone: Zone number
        substation_id: Optional substation or region string identifier
    """

    id: Union[int, str]
    bus_type: int = 1
    v_mag: float = 1.0
    v_angle: float = 0.0
    p_mismatch: float = 0.0
    q_mismatch: float = 0.0
    base_kv: float = 100.0
    bus_area: int = 1
    zone: int = 1
    substation_id: Optional[str] = None


@dataclass
class BusEnergyBalance:
    """Detailed energy balance container for a single bus.

    Maintains all connected injections and withdrawals:
    - outgoing_flows: Branch flows leaving this bus at the 'from' end (MW, MVAr)
    - incoming_flows: Branch flows entering this bus at the 'to' end (MW, MVAr)
    - shunts: Shunt injections/withdrawals (MW, MVAr)
    - facts: FACTS / Static Var Compensators injections (MW, MVAr)
    - loads: Active and reactive power demand (MW, MVAr)
    - generations: Generator output (MW, MVAr)
    - hvdc: HVDC / DC line injections (MW, MVAr)

    Attributes:
        bus_id: Bus identifier
        vm: Voltage magnitude (p.u.)
        va: Voltage angle (degrees)
        bus_type: MATPOWER bus type (1=PQ, 2=PV, 3=Slack, 4=Isolated)
        base_kv: Base voltage (kV)
        bus_area: Area number
        zone: Loss zone
        connected_component: Connected component index
        synchronous_component: Synchronous component index
    """

    bus_id: Union[int, str] = 0
    vm: float = 1.0
    va: float = 0.0
    bus_type: int = 1
    base_kv: float = 100.0
    bus_area: int = 1
    zone: int = 1
    connected_component: int = 0
    synchronous_component: int = 0
    outgoing_flows: Dict[Any, Flow] = field(default_factory=dict)
    incoming_flows: Dict[Any, Flow] = field(default_factory=dict)
    shunts: Dict[Any, Flow] = field(default_factory=dict)
    facts: Dict[Any, Flow] = field(default_factory=dict)
    loads: Dict[Any, Flow] = field(default_factory=dict)
    generations: Dict[Any, Flow] = field(default_factory=dict)
    hvdc: Dict[Any, Flow] = field(default_factory=dict)

    def total_outgoing_flow(self) -> Flow:
        """Sum of outgoing branch flows leaving this bus."""
        p = sum(f.p for f in self.outgoing_flows.values())
        q = sum(f.q for f in self.outgoing_flows.values())
        return Flow(p, q)

    def total_incoming_flow(self) -> Flow:
        """Sum of incoming branch flows entering this bus."""
        p = sum(f.p for f in self.incoming_flows.values())
        q = sum(f.q for f in self.incoming_flows.values())
        return Flow(p, q)

    def total_gen(self) -> Flow:
        """Sum of generator generation at this bus."""
        p = sum(f.p for f in self.generations.values())
        q = sum(f.q for f in self.generations.values())
        return Flow(p, q)

    def total_load(self) -> Flow:
        """Sum of loads consumed at this bus."""
        p = sum(f.p for f in self.loads.values())
        q = sum(f.q for f in self.loads.values())
        return Flow(p, q)

    def total_shunt(self) -> Flow:
        """Sum of shunt flows at this bus."""
        p = sum(f.p for f in self.shunts.values())
        q = sum(f.q for f in self.shunts.values())
        return Flow(p, q)

    def total_facts(self) -> Flow:
        """Sum of FACTS flows at this bus."""
        p = sum(f.p for f in self.facts.values())
        q = sum(f.q for f in self.facts.values())
        return Flow(p, q)

    def total_hvdc(self) -> Flow:
        """Sum of HVDC flows at this bus."""
        p = sum(f.p for f in self.hvdc.values())
        q = sum(f.q for f in self.hvdc.values())
        return Flow(p, q)

    def balance(self) -> Flow:
        """Calculate and return net power mismatch at this bus."""
        return calc_bus_balance(self)


# =============================================================================
# Branch and Line Flow Calculations (MATPOWER Model)
# =============================================================================


def calc_flow_into_branch(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b: float = 0.0,
    shift: float = 0.0,
    base_mva: float = 100.0,
) -> Flow:
    """Calculate active and reactive power flow INTO a branch at the 'from' bus.

    Supports transmission lines and transformers with off-nominal tap ratio
    and phase shift angle (tap is placed at the 'from' bus side as in MATPOWER).

    Args:
        vsm: Voltage magnitude at 'from' (sending) bus (p.u.)
        vrm: Voltage magnitude at 'to' (receiving) bus (p.u.)
        vsa: Voltage angle at 'from' bus (degrees)
        vra: Voltage angle at 'to' bus (degrees)
        r: Series resistance (p.u.)
        x: Series reactance (p.u.)
        b: Total branch charging susceptance (p.u.)
        shift: Transformer phase shift angle (degrees, positive = delay)
        base_mva: System base MVA (default: 100.0)

    Returns:
        Flow(p, q) in MW and MVAr leaving the 'from' bus into the branch.
    """
    shift_rad = np.deg2rad(shift)
    delta = np.deg2rad(vsa) - np.deg2rad(vra) - shift_rad
    # r = 0  # TODO
    z_square = r**2 + x**2

    if z_square == 0.0:
        return Flow(0.0, 0.0)

    # Active power flow into branch at from bus
    ps = (vsm / z_square) * (
        vsm * r - r * vrm * np.cos(delta) + x * vrm * np.sin(delta)
    )

    # Reactive power flow into branch at from bus
    qs = (vsm / z_square) * (
        vsm * x
        - x * vrm * np.cos(delta)
        + r * vrm * np.sin(delta)
        - vsm * z_square * (b / 2.0)
    )

    return Flow(float(ps * base_mva), float(qs * base_mva))


def calc_flow_from_branch(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b: float = 0.0,
    shift: float = 0.0,
    base_mva: float = 100.0,
) -> Flow:
    """Calculate active and reactive power flow entering the 'to' bus FROM the branch.

    Args:
        vsm: Voltage magnitude at 'from' (sending) bus (p.u.)
        vrm: Voltage magnitude at 'to' (receiving) bus (p.u.)
        vsa: Voltage angle at 'from' bus (degrees)
        vra: Voltage angle at 'to' bus (degrees)
        r: Series resistance (p.u.)
        x: Series reactance (p.u.)
        b: Total branch charging susceptance (p.u.)
        shift: Transformer phase shift angle (degrees, positive = delay)
        base_mva: System base MVA (default: 100.0)

    Returns:
        Flow(p, q) in MW and MVAr entering the 'to' bus from the branch.
    """
    # r = 0  # TODO
    shift_rad = np.deg2rad(shift)
    delta = np.deg2rad(vsa) - np.deg2rad(vra) - shift_rad
    z_square = r**2 + x**2

    if z_square == 0.0:
        return Flow(0.0, 0.0)

    # Active power flow entering receiving bus from branch
    pr = (vrm / z_square) * (
        -vrm * r + r * vsm * np.cos(delta) + x * vsm * np.sin(delta)
    )

    # Reactive power flow entering receiving bus from branch
    qr = (vrm / z_square) * (
        -vrm * x
        + x * vsm * np.cos(delta)
        - r * vsm * np.sin(delta)
        + vrm * z_square * (b / 2.0)
    )

    return Flow(float(pr * base_mva), float(qr * base_mva))


def calc_flow_into_line(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b: float,
    base_mva: float = 100.0,
) -> Flow:
    """Calculate flow into a transmission line at the sending bus (psbl.py compatibility)."""
    return calc_flow_into_branch(
        vsm, vrm, vsa, vra, r, x, b, tap=0.0, shift=0.0, base_mva=base_mva
    )


def calc_flow_from_line(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b: float,
    base_mva: float = 100.0,
) -> Flow:
    """Calculate flow from a transmission line at the receiving bus (psbl.py compatibility)."""
    return calc_flow_from_branch(
        vsm, vrm, vsa, vra, r, x, b, shift=0.0, base_mva=base_mva
    )


# =============================================================================
# Bus Balance Calculation
# =============================================================================


def calc_bus_balance(bus: BusEnergyBalance) -> Flow:
    """Calculate net active and reactive power mismatch (imbalance) at a bus.

    Follows the psbl.py nodal balance formulation:
    Mismatch = (Power Leaving Bus) - (Power Injected into Bus)
             = Sum(Outgoing Flows) - Sum(Incoming Flows) + Loads - Generations - Shunts - FACTS + HVDC

    When the power flow solution is exact, calc_bus_balance returns Flow(0, 0).

    Args:
        bus: BusEnergyBalance instance containing all flows, loads, gens, and shunts.

    Returns:
        Flow(p_mismatch, q_mismatch) in MW and MVAr.
    """
    p, q = 0.0, 0.0

    # Outgoing branch flows (leaving bus)
    for flow in bus.outgoing_flows.values():
        p += flow.p
        q += flow.q

    # Incoming branch flows (entering bus, subtracted from power leaving)
    for flow in bus.incoming_flows.values():
        p -= flow.p
        q -= flow.q

    # Shunt injections (injected power reduces deficit, so subtracted)
    for flow in bus.shunts.values():
        p -= flow.p
        q -= flow.q

    # FACTS / SVC injections
    for flow in bus.facts.values():
        p -= flow.p
        q -= flow.q

    # Loads (power demanded / leaving bus)
    for flow in bus.loads.values():
        p += flow.p
        q += flow.q

    # Generations (power generated / entering bus)
    for flow in bus.generations.values():
        p -= flow.p
        q -= flow.q

    # HVDC line flows
    for flow in bus.hvdc.values():
        p += flow.p
        q += flow.q

    return Flow(p, q)


# =============================================================================
# Component Extractors for MATPOWER Data Structures
# =============================================================================


def _get_case_components(case: Any) -> Tuple[float, Any, Any, Any]:
    """Helper to extract (baseMVA, bus, gen, branch) from any MATPOWER structure."""
    if hasattr(case, "baseMVA") and hasattr(case, "bus") and hasattr(case, "gen"):
        # MatpowerCase object
        return float(case.baseMVA), case.bus, case.gen, case.branch
    elif isinstance(case, dict):
        base_mva = float(case.get("baseMVA", 100.0))
        return (
            base_mva,
            case.get("bus", []),
            case.get("gen", []),
            case.get("branch", []),
        )
    raise ValueError(f"Unsupported MATPOWER case type: {type(case)}")


def init_bus_energy_balances(case: Any) -> Dict[Union[int, str], BusEnergyBalance]:
    """Initialize BusEnergyBalance dictionary from a MATPOWER case.

    Args:
        case: MatpowerCase object, MATPOWERCase dict, or standard case dictionary.

    Returns:
        Dict mapping bus_id -> BusEnergyBalance initialized with voltage and bus properties.
    """
    _, bus_data, _, _ = _get_case_components(case)
    id_to_bus: Dict[Union[int, str], BusEnergyBalance] = {}

    if isinstance(bus_data, np.ndarray):
        for idx in range(bus_data.shape[0]):
            row = bus_data[idx]
            b_id = int(row[BUS_I])
            id_to_bus[b_id] = BusEnergyBalance(
                bus_id=b_id,
                vm=float(row[VM]),
                va=float(row[VA]),
                bus_type=int(row[BUS_TYPE]),
                base_kv=float(row[BASE_KV]) if len(row) > BASE_KV else 100.0,
                bus_area=int(row[BUS_AREA]) if len(row) > BUS_AREA else 1,
                zone=int(row[ZONE]) if len(row) > ZONE else 1,
            )
    else:
        for idx, b in enumerate(bus_data):
            if hasattr(b, "bus_i"):
                # Bus dataclass instance
                id_to_bus[b.bus_i] = BusEnergyBalance(
                    bus_id=b.bus_i,
                    vm=b.vm,
                    va=b.va,
                    bus_type=b.bus_type,
                    base_kv=b.base_kv,
                    bus_area=b.bus_area,
                    zone=b.zone,
                )
            elif isinstance(b, (list, tuple)):
                b_id = int(b[BUS_I])
                id_to_bus[b_id] = BusEnergyBalance(
                    bus_id=b_id,
                    vm=float(b[VM]),
                    va=float(b[VA]),
                    bus_type=int(b[BUS_TYPE]),
                    base_kv=float(b[BASE_KV]) if len(b) > BASE_KV else 100.0,
                    bus_area=int(b[BUS_AREA]) if len(b) > BUS_AREA else 1,
                    zone=int(b[ZONE]) if len(b) > ZONE else 1,
                )

    return id_to_bus


def get_bus_generations(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Populate generator active and reactive generation into id_to_bus.

    Args:
        case: MATPOWER case object or dictionary.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.

    Returns:
        Updated id_to_bus dictionary.
    """
    _, _, gen_data, _ = _get_case_components(case)

    if isinstance(gen_data, np.ndarray):
        for idx in range(gen_data.shape[0]):
            row = gen_data[idx]
            status = int(row[GEN_STATUS]) if len(row) > GEN_STATUS else 1
            if status > 0:
                gen_bus = int(row[GEN_BUS])
                if gen_bus in id_to_bus:
                    pg = float(row[PG])
                    qg = float(row[QG])
                    id_to_bus[gen_bus].generations[f"gen_{idx}"] = Flow(pg, qg)
    else:
        for idx, g in enumerate(gen_data):
            if hasattr(g, "gen_bus"):
                if g.is_in_service and g.gen_bus in id_to_bus:
                    id_to_bus[g.gen_bus].generations[f"gen_{idx}"] = Flow(g.pg, g.qg)
            elif isinstance(g, (list, tuple)):
                status = int(g[GEN_STATUS]) if len(g) > GEN_STATUS else 1
                if status > 0:
                    gen_bus = int(g[GEN_BUS])
                    if gen_bus in id_to_bus:
                        id_to_bus[gen_bus].generations[f"gen_{idx}"] = Flow(
                            float(g[PG]), float(g[QG])
                        )

    return id_to_bus


def get_bus_loads(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Populate active and reactive power loads into id_to_bus.

    Args:
        case: MATPOWER case object or dictionary.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.

    Returns:
        Updated id_to_bus dictionary.
    """
    _, bus_data, _, _ = _get_case_components(case)

    if isinstance(bus_data, np.ndarray):
        for idx in range(bus_data.shape[0]):
            row = bus_data[idx]
            b_id = int(row[BUS_I])
            b_type = int(row[BUS_TYPE])
            if b_type != 4 and b_id in id_to_bus:  # Skip isolated buses
                pd = float(row[PD])
                qd = float(row[QD])
                if pd != 0.0 or qd != 0.0:
                    id_to_bus[b_id].loads[f"load_{idx}"] = Flow(pd, qd)
    else:
        for idx, b in enumerate(bus_data):
            if hasattr(b, "bus_i"):
                if b.is_in_service and b.bus_i in id_to_bus:
                    if b.pd != 0.0 or b.qd != 0.0:
                        id_to_bus[b.bus_i].loads[f"load_{idx}"] = Flow(b.pd, b.qd)
            elif isinstance(b, (list, tuple)):
                b_id = int(b[BUS_I])
                b_type = int(b[BUS_TYPE])
                if b_type != 4 and b_id in id_to_bus:
                    pd = float(b[PD])
                    qd = float(b[QD])
                    if pd != 0.0 or qd != 0.0:
                        id_to_bus[b_id].loads[f"load_{idx}"] = Flow(pd, qd)

    return id_to_bus


def get_shunts(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Populate shunt active and reactive power flows into id_to_bus.

    In MATPOWER bus format:
    - gs: Shunt conductance (MW demanded at V = 1.0 p.u.)
    - bs: Shunt susceptance (MVAr injected at V = 1.0 p.u.)

    At voltage Vm:
    - Active power injected: P_sh = -gs * Vm^2
    - Reactive power injected: Q_sh = bs * Vm^2

    Args:
        case: MATPOWER case object or dictionary.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.

    Returns:
        Updated id_to_bus dictionary.
    """
    _, bus_data, _, _ = _get_case_components(case)

    if isinstance(bus_data, np.ndarray):
        for idx in range(bus_data.shape[0]):
            row = bus_data[idx]
            b_id = int(row[BUS_I])
            b_type = int(row[BUS_TYPE])
            if b_type != 4 and b_id in id_to_bus:
                gs = float(row[GS])
                bs = float(row[BS])
                if gs != 0.0 or bs != 0.0:
                    vm = id_to_bus[b_id].vm
                    # P injected is -gs * vm^2, Q injected is bs * vm^2
                    p_inj = -gs * (vm**2)
                    q_inj = bs * (vm**2)
                    id_to_bus[b_id].shunts[f"shunt_{idx}"] = Flow(p_inj, q_inj)
    else:
        for idx, b in enumerate(bus_data):
            if hasattr(b, "bus_i"):
                if b.is_in_service and b.bus_i in id_to_bus:
                    if b.gs != 0.0 or b.bs != 0.0:
                        vm = id_to_bus[b.bus_i].vm
                        p_inj = -b.gs * (vm**2)
                        q_inj = b.bs * (vm**2)
                        id_to_bus[b.bus_i].shunts[f"shunt_{idx}"] = Flow(p_inj, q_inj)
            elif isinstance(b, (list, tuple)):
                b_id = int(b[BUS_I])
                b_type = int(b[BUS_TYPE])
                if b_type != 4 and b_id in id_to_bus:
                    gs = float(b[GS])
                    bs = float(b[BS])
                    if gs != 0.0 or bs != 0.0:
                        vm = id_to_bus[b_id].vm
                        p_inj = -gs * (vm**2)
                        q_inj = bs * (vm**2)
                        id_to_bus[b_id].shunts[f"shunt_{idx}"] = Flow(p_inj, q_inj)

    return id_to_bus


def get_branch_flows(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Calculate and populate all branch (lines + transformers) flows into id_to_bus.

    Args:
        case: MATPOWER case object or dictionary.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.

    Returns:
        Updated id_to_bus dictionary with outgoing and incoming branch flows.
    """
    base_mva, _, _, branch_data = _get_case_components(case)

    if isinstance(branch_data, np.ndarray):
        for idx in range(branch_data.shape[0]):
            row = branch_data[idx]
            status = int(row[10]) if len(row) > 10 else 1
            if status <= 0:
                continue

            fb = int(row[F_BUS])
            tb = int(row[T_BUS])
            if fb not in id_to_bus or tb not in id_to_bus:
                continue

            r = float(row[2])
            x = float(row[3])
            b = float(row[4])
            tap = float(row[TAP]) if len(row) > TAP else 0.0
            shift = float(row[SHIFT]) if len(row) > SHIFT else 0.0

            vsm = id_to_bus[fb].vm
            vsa = id_to_bus[fb].va
            vrm = id_to_bus[tb].vm
            vra = id_to_bus[tb].va

            flow_into = calc_flow_into_branch(
                vsm, vrm, vsa, vra, r, x, b, shift, base_mva
            )
            flow_from = calc_flow_from_branch(
                vsm, vrm, vsa, vra, r, x, b, shift, base_mva
            )

            id_to_bus[fb].outgoing_flows[f"br_{idx}"] = flow_into
            id_to_bus[tb].incoming_flows[f"br_{idx}"] = flow_from

    else:
        for idx, br in enumerate(branch_data):
            if hasattr(br, "f_bus"):
                if not br.is_in_service:
                    continue
                fb = br.f_bus
                tb = br.t_bus
                if fb not in id_to_bus or tb not in id_to_bus:
                    continue

                vsm = id_to_bus[fb].vm
                vsa = id_to_bus[fb].va
                vrm = id_to_bus[tb].vm
                vra = id_to_bus[tb].va

                flow_into = calc_flow_into_branch(
                    vsm,
                    vrm,
                    vsa,
                    vra,
                    br.br_r,
                    br.br_x,
                    br.br_b,
                    br.shift,
                    base_mva,
                )
                flow_from = calc_flow_from_branch(
                    vsm,
                    vrm,
                    vsa,
                    vra,
                    br.br_r,
                    br.br_x,
                    br.br_b,
                    br.shift,
                    base_mva,
                )

                id_to_bus[fb].outgoing_flows[f"br_{idx}"] = flow_into
                id_to_bus[tb].incoming_flows[f"br_{idx}"] = flow_from

            elif isinstance(br, (list, tuple)):
                status = int(br[10]) if len(br) > 10 else 1
                if status <= 0:
                    continue

                fb = int(br[F_BUS])
                tb = int(br[T_BUS])
                if fb not in id_to_bus or tb not in id_to_bus:
                    continue

                r = float(br[2])
                x = float(br[3])
                b = float(br[4])
                tap = float(br[TAP]) if len(br) > TAP else 0.0
                shift = float(br[SHIFT]) if len(br) > SHIFT else 0.0

                vsm = id_to_bus[fb].vm
                vsa = id_to_bus[fb].va
                vrm = id_to_bus[tb].vm
                vra = id_to_bus[tb].va

                flow_into = calc_flow_into_branch(
                    vsm, vrm, vsa, vra, r, x, b, shift, base_mva
                )
                flow_from = calc_flow_from_branch(
                    vsm, vrm, vsa, vra, r, x, b, shift, base_mva
                )

                id_to_bus[fb].outgoing_flows[f"br_{idx}"] = flow_into
                id_to_bus[tb].incoming_flows[f"br_{idx}"] = flow_from

    return id_to_bus


def get_line_flows(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Compatibility alias for get_branch_flows."""
    return get_branch_flows(case, id_to_bus)


def get_two_winds_trans(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Compatibility alias for get_branch_flows (MATPOWER branches contain 2W transformers)."""
    return get_branch_flows(case, id_to_bus)


def get_three_winds_trans(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Compatibility helper (in MATPOWER format, 3W transformers are converted to star bus branches)."""
    return id_to_bus


def get_hvdc_flow(
    case: Any, id_to_bus: Dict[Union[int, str], BusEnergyBalance]
) -> Dict[Union[int, str], BusEnergyBalance]:
    """Extract HVDC / DC line flows if 'dcline' matrix is present in MATPOWER case."""
    if isinstance(case, dict) and "dcline" in case:
        dclines = case["dcline"]
        if isinstance(dclines, np.ndarray) and dclines.size > 0:
            for idx in range(dclines.shape[0]):
                row = dclines[idx]
                status = int(row[2]) if len(row) > 2 else 1
                if status <= 0:
                    continue
                fb = int(row[0])
                tb = int(row[1])
                pf = float(row[3]) if len(row) > 3 else 0.0
                pt = float(row[4]) if len(row) > 4 else 0.0
                qf = float(row[5]) if len(row) > 5 else 0.0
                qt = float(row[6]) if len(row) > 6 else 0.0

                if fb in id_to_bus:
                    id_to_bus[fb].hvdc[f"dcline_{idx}"] = Flow(pf, qf)
                if tb in id_to_bus:
                    id_to_bus[tb].hvdc[f"dcline_{idx}"] = Flow(pt, qt)

    return id_to_bus


def build_bus_energy_balances(case: Any) -> Dict[Union[int, str], BusEnergyBalance]:
    """Construct complete BusEnergyBalance mapping for all buses in a MATPOWER case.

    Executes full pipeline:
    1. Initializing bus structures and voltages
    2. Extracting generator power
    3. Extracting load demand
    4. Extracting shunt injections
    5. Computing branch flows for all lines and transformers
    6. Extracting DC lines / HVDC flows

    Args:
        case: MatpowerCase object, dict, or MATPOWERCase.

    Returns:
        Dict mapping bus_id -> BusEnergyBalance.
    """
    id_to_bus = init_bus_energy_balances(case)
    id_to_bus = get_bus_generations(case, id_to_bus)
    id_to_bus = get_bus_loads(case, id_to_bus)
    id_to_bus = get_shunts(case, id_to_bus)
    id_to_bus = get_branch_flows(case, id_to_bus)
    id_to_bus = get_hvdc_flow(case, id_to_bus)
    return id_to_bus


# =============================================================================
# Low Impedance, Short Circuit, and Islanding Tools (following psbl.py)
# =============================================================================


def is_buses_short_circuited(
    bus1: BusEnergyBalance,
    bus2: BusEnergyBalance,
    vm_tol: float = 0.01,
    va_tol: float = 0.01,
) -> bool:
    """Check if two buses are short-circuited (nearly identical voltage magnitude & angle)."""
    return abs(bus1.vm - bus2.vm) < vm_tol and abs(bus1.va - bus2.va) < va_tol


def open_low_impedance_branches(
    case: Any,
    id_to_bus: Dict[Union[int, str], BusEnergyBalance],
    x_threshold: float = 0.0002,
    angle_diff_ratio: float = 30.0,
    min_va_diff: float = 0.01,
) -> List[int]:
    """Identify branches with low reactance and high angle differences to disconnect.

    Args:
        case: MATPOWER case object or dict.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.
        x_threshold: Reactance threshold (default: 0.0002 p.u.).
        angle_diff_ratio: Angle diff (rad) / x ratio threshold (default: 30.0).
        min_va_diff: Minimum angle difference in degrees (default: 0.01 deg).

    Returns:
        List of branch indices that were identified for disconnection.
    """
    _, _, _, branch_data = _get_case_components(case)
    disconnected_indices: List[int] = []

    if isinstance(branch_data, np.ndarray):
        for idx in range(branch_data.shape[0]):
            row = branch_data[idx]
            status = int(row[10]) if len(row) > 10 else 1
            if status <= 0:
                continue

            fb = int(row[F_BUS])
            tb = int(row[T_BUS])
            x = float(row[3])

            if fb in id_to_bus and tb in id_to_bus and abs(x) < x_threshold:
                bus1 = id_to_bus[fb]
                bus2 = id_to_bus[tb]
                va_diff_rad = abs(np.radians(bus1.va) - np.radians(bus2.va))
                va_diff_deg = abs(bus1.va - bus2.va)

                if (
                    va_diff_rad / max(abs(x), 1e-6)
                ) > angle_diff_ratio and va_diff_deg > min_va_diff:
                    disconnected_indices.append(idx)
                    row[10] = 0  # Disconnect in numpy matrix

    elif isinstance(branch_data, list):
        for idx, br in enumerate(branch_data):
            if hasattr(br, "br_status"):
                if not br.is_in_service:
                    continue
                fb = br.f_bus
                tb = br.t_bus
                x = br.br_x
                if fb in id_to_bus and tb in id_to_bus and abs(x) < x_threshold:
                    bus1 = id_to_bus[fb]
                    bus2 = id_to_bus[tb]
                    va_diff_rad = abs(np.radians(bus1.va) - np.radians(bus2.va))
                    va_diff_deg = abs(bus1.va - bus2.va)
                    if (
                        va_diff_rad / max(abs(x), 1e-6)
                    ) > angle_diff_ratio and va_diff_deg > min_va_diff:
                        disconnected_indices.append(idx)
                        br.br_status = 0
            elif isinstance(br, list):
                status = int(br[10]) if len(br) > 10 else 1
                if status <= 0:
                    continue
                fb = int(br[F_BUS])
                tb = int(br[T_BUS])
                x = float(br[3])
                if fb in id_to_bus and tb in id_to_bus and abs(x) < x_threshold:
                    bus1 = id_to_bus[fb]
                    bus2 = id_to_bus[tb]
                    va_diff_rad = abs(np.radians(bus1.va) - np.radians(bus2.va))
                    va_diff_deg = abs(bus1.va - bus2.va)
                    if (
                        va_diff_rad / max(abs(x), 1e-6)
                    ) > angle_diff_ratio and va_diff_deg > min_va_diff:
                        disconnected_indices.append(idx)
                        br[10] = 0

    return disconnected_indices


def find_short_circuit_buses(
    case: Any,
    id_to_bus: Dict[Union[int, str], BusEnergyBalance],
    vm_tol: float = 0.01,
    va_tol: float = 0.01,
) -> Tuple[List[int], List[Set[Union[int, str]]]]:
    """Find short-circuited connected components across branches.

    Args:
        case: MATPOWER case object or dict.
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.
        vm_tol: Voltage magnitude tolerance (p.u.).
        va_tol: Voltage angle tolerance (degrees).

    Returns:
        Tuple of (disconnected branch indices, list of connected component sets of bus IDs).
    """
    try:
        import networkx as nx
    except ImportError:
        # Fallback simple BFS if networkx is not installed
        return [], []

    _, _, _, branch_data = _get_case_components(case)
    g = nx.Graph()
    disconnected_branches: List[int] = []

    if isinstance(branch_data, np.ndarray):
        for idx in range(branch_data.shape[0]):
            row = branch_data[idx]
            status = int(row[10]) if len(row) > 10 else 1
            if status <= 0:
                continue
            fb = int(row[F_BUS])
            tb = int(row[T_BUS])
            if fb in id_to_bus and tb in id_to_bus:
                if is_buses_short_circuited(
                    id_to_bus[fb], id_to_bus[tb], vm_tol, va_tol
                ):
                    g.add_edge(fb, tb)
                    disconnected_branches.append(idx)
                    row[10] = 0
    else:
        for idx, br in enumerate(branch_data):
            if hasattr(br, "f_bus"):
                if not br.is_in_service:
                    continue
                fb, tb = br.f_bus, br.t_bus
                if fb in id_to_bus and tb in id_to_bus:
                    if is_buses_short_circuited(
                        id_to_bus[fb], id_to_bus[tb], vm_tol, va_tol
                    ):
                        g.add_edge(fb, tb)
                        disconnected_branches.append(idx)
                        br.br_status = 0
            elif isinstance(br, (list, tuple)):
                status = int(br[10]) if len(br) > 10 else 1
                if status <= 0:
                    continue
                fb, tb = int(br[F_BUS]), int(br[T_BUS])
                if fb in id_to_bus and tb in id_to_bus:
                    if is_buses_short_circuited(
                        id_to_bus[fb], id_to_bus[tb], vm_tol, va_tol
                    ):
                        g.add_edge(fb, tb)
                        disconnected_branches.append(idx)
                        if isinstance(br, list):
                            br[10] = 0

    components = [set(c) for c in nx.connected_components(g)]
    return disconnected_branches, components


def get_equivalent_buses(
    islands: Sequence[Set[Union[int, str]]],
    star_buses: Optional[Set[Union[int, str]]] = None,
) -> Dict[Union[int, str], Union[int, str]]:
    """Map each bus in an island to a single retained representative bus."""
    if star_buses is None:
        star_buses = set()

    result: Dict[Union[int, str], Union[int, str]] = {}
    for island in islands:
        island_list = list(island)
        bus_to_keep = island_list[0]
        for b in island:
            if b in star_buses:
                bus_to_keep = b
                break

        for b in island:
            if b != bus_to_keep:
                result[b] = bus_to_keep

    return result


def subs_with_volt_violations(
    case_or_id_to_bus: Union[Any, Dict[Union[int, str], BusEnergyBalance]],
    top_n: int = 10,
    vmin: float = 0.9,
    vmax: float = 1.1,
    violate_upper_limit: bool = True,
) -> List[BusInfo]:
    """Find buses with top voltage magnitude limit violations (following psbl.py).

    Args:
        case_or_id_to_bus: MATPOWER case or id_to_bus dictionary.
        top_n: Number of top violating buses to return.
        vmin: Minimum voltage limit (p.u.).
        vmax: Maximum voltage limit (p.u.).
        violate_upper_limit: If True, sort by highest voltage (overvoltage).
                             If False, sort by lowest voltage (undervoltage).

    Returns:
        List of BusInfo instances.
    """
    if isinstance(case_or_id_to_bus, dict) and all(
        isinstance(v, BusEnergyBalance) for v in case_or_id_to_bus.values()
    ):
        id_to_bus = case_or_id_to_bus
    else:
        id_to_bus = init_bus_energy_balances(case_or_id_to_bus)

    buses = [b for b in id_to_bus.values() if b.bus_type != 4]

    if violate_upper_limit:
        buses = [b for b in buses if b.vm > vmax]
        buses.sort(key=lambda b: b.vm, reverse=True)
    else:
        buses = [b for b in buses if b.vm < vmin]
        buses.sort(key=lambda b: b.vm, reverse=False)

    result: List[BusInfo] = []
    for b in buses[:top_n]:
        bal = calc_bus_balance(b)
        result.append(
            BusInfo(
                id=b.bus_id,
                bus_type=b.bus_type,
                v_mag=b.vm,
                v_angle=b.va,
                p_mismatch=bal.p,
                q_mismatch=bal.q,
                base_kv=b.base_kv,
                bus_area=b.bus_area,
                zone=b.zone,
            )
        )
    return result


# =============================================================================
# Validation Report and High-Level Validation Functions
# =============================================================================


@dataclass
class ValidationReport:
    """Comprehensive validation report for MATPOWER energy balance."""

    is_balanced: bool
    num_buses: int
    num_generators: int
    num_branches: int
    max_p_mismatch: float
    max_q_mismatch: float
    rms_p_mismatch: float
    rms_q_mismatch: float
    max_p_bus: Union[int, str]
    max_q_bus: Union[int, str]
    total_p_gen: float
    total_q_gen: float
    total_p_load: float
    total_q_load: float
    total_p_shunt: float
    total_q_shunt: float
    total_p_loss: float
    total_q_loss: float
    tolerance_p: float
    tolerance_q: float
    num_violations: int
    violations: List[BusInfo] = field(default_factory=list)

    def summary(self) -> str:
        """Generate formatted summary string for validation report."""
        status_str = (
            "PASSED [BALANCED]" if self.is_balanced else "FAILED [MISMATCH DETECTED]"
        )
        lines = [
            "=" * 70,
            f"{'MATPOWER BUS ENERGY BALANCE VALIDATION REPORT':^70}",
            "=" * 70,
            f"  Status:                   {status_str}",
            f"  Buses Analyzed:           {self.num_buses:,d}",
            f"  In-Service Generators:    {self.num_generators:,d}",
            f"  In-Service Branches:      {self.num_branches:,d}",
            "-" * 70,
            "  POWER MISMATCH SUMMARY:",
            f"    Max |P| Mismatch:       {self.max_p_mismatch:>14.6f} MW   (at Bus {self.max_p_bus})",
            f"    Max |Q| Mismatch:       {self.max_q_mismatch:>14.6f} MVAr (at Bus {self.max_q_bus})",
            f"    RMS P Mismatch:         {self.rms_p_mismatch:>14.6f} MW",
            f"    RMS Q Mismatch:         {self.rms_q_mismatch:>14.6f} MVAr",
            f"    P / Q Tolerance:        {self.tolerance_p:.4f} MW / {self.tolerance_q:.4f} MVAr",
            f"    Tolerance Violations:   {self.num_violations:,d} buses",
            "-" * 70,
            "  SYSTEM TOTALS:",
            f"    Total Generation:       {self.total_p_gen:>14.4f} MW  | {self.total_q_gen:>14.4f} MVAr",
            f"    Total Load:             {self.total_p_load:>14.4f} MW  | {self.total_q_load:>14.4f} MVAr",
            f"    Total Shunt Injection:  {self.total_p_shunt:>14.4f} MW  | {self.total_q_shunt:>14.4f} MVAr",
            f"    Total Branch Losses:    {self.total_p_loss:>14.4f} MW  | {self.total_q_loss:>14.4f} MVAr",
            "=" * 70,
        ]
        return "\n".join(lines)


def find_bus_balance_violations(
    id_to_bus: Dict[Union[int, str], BusEnergyBalance],
    p_tol: float = 1e-3,
    q_tol: float = 1e-3,
    top_n: Optional[int] = 10,
    sort_by: str = "p",
) -> List[BusInfo]:
    """Find all buses whose power mismatch exceeds specified tolerances.

    Args:
        id_to_bus: Dict mapping bus_id -> BusEnergyBalance.
        p_tol: Active power mismatch tolerance (MW).
        q_tol: Reactive power mismatch tolerance (MVAr).
        top_n: Maximum number of records to return (None for all).
        sort_by: Sorting criterion ('p' for |P|, 'q' for |Q|, 's' for apparent mismatch |S|).

    Returns:
        List of BusInfo instances sorted descending by mismatch magnitude.
    """
    violations: List[BusInfo] = []

    for b in id_to_bus.values():
        if b.bus_type == 4:  # Skip isolated buses
            continue
        bal = calc_bus_balance(b)
        if abs(bal.p) > p_tol or abs(bal.q) > q_tol:
            violations.append(
                BusInfo(
                    id=b.bus_id,
                    bus_type=b.bus_type,
                    v_mag=b.vm,
                    v_angle=b.va,
                    p_mismatch=bal.p,
                    q_mismatch=bal.q,
                    base_kv=b.base_kv,
                    bus_area=b.bus_area,
                    zone=b.zone,
                )
            )

    if sort_by.lower() == "q":
        violations.sort(key=lambda x: abs(x.q_mismatch), reverse=True)
    elif sort_by.lower() == "s":
        violations.sort(
            key=lambda x: math.sqrt(x.p_mismatch**2 + x.q_mismatch**2), reverse=True
        )
    else:
        violations.sort(key=lambda x: abs(x.p_mismatch), reverse=True)

    if top_n is not None:
        return violations[:top_n]
    return violations


def print_bus_balance_summary(
    id_to_bus: Dict[Union[int, str], BusEnergyBalance],
    top_n: int = 10,
    p_tol: float = 1e-3,
    q_tol: float = 1e-3,
) -> None:
    """Print an interactive table of the top mismatched buses."""
    violations = find_bus_balance_violations(
        id_to_bus, p_tol=p_tol, q_tol=q_tol, top_n=top_n, sort_by="p"
    )

    print("\n" + "=" * 140)
    print(f"{'TOP BUS ENERGY BALANCE MISMATCHES':^140}")
    print("=" * 140)
    print(
        f"{'Bus ID':>10} | {'Type':<6} | {'Vm (pu)':>7} | {'Va (deg)':>8} | {'P Mismatch (MW)':>16} | {'Q Mismatch (MVAr)':>18} | {'Flow':>10} | {'Gen':>10} | {'Load':>10} | {'Shunt':>10} | {'Facts':>10} | {'Hvdc':>10}"
    )
    print("-" * 140)

    type_map = {1: "PQ", 2: "PV", 3: "Slack", 4: "Isol"}
    for v in violations:
        t_str = type_map.get(v.bus_type, str(v.bus_type))
        bus = id_to_bus[v.id]
        flow = sum([f.p for f in bus.incoming_flows.values()]) - sum(
            [f.p for f in bus.outgoing_flows.values()]
        )
        gen = sum([f.p for f in bus.generations.values()])
        load = sum([f.p for f in bus.loads.values()])
        shunt = sum([f.p for f in bus.shunts.values()])
        facts = sum([f.p for f in bus.facts.values()])
        hvdc = sum([f.p for f in bus.hvdc.values()])
        print(
            f"{v.id:>10} | {t_str:<6} | {v.v_mag:>7.4f} | {v.v_angle:>8.2f} | {v.p_mismatch:>16.4f} | {v.q_mismatch:>18.4f} | {flow:>10.0f} | {gen:>10.0f} | {load:>10.0f} | {shunt:>10.0f} | {facts:>10.0f} | {hvdc:>10.0f}"
        )
    print("=" * 140 + "\n")


def print_branch_large_flows(
    id_to_bus: Dict[Union[int, str], BusEnergyBalance], mpc, top_n: int = 5
):
    branches = {}
    for bus in id_to_bus.values():
        for br, flow in bus.incoming_flows.items():
            branches[br] = abs(flow.p)
        for br, flow in bus.outgoing_flows.items():
            branches[br] = abs(flow.p)
    branches = dict(sorted(branches.items(), key=lambda x: x[1], reverse=True))
    print("\n" + "=" * 120)
    print(f"{'TOP BRANCHES WITH LARGEST POWER FLOWS':^80}")
    print("=" * 120)
    print(
        f"{'From Bus':>10} | {'To Bus':>10} | {'R':>10} | {'X':>10} | {'P Flow (MW)':>16} | {'From Va':>10} | {'To Va':>10} | {'Rate A (MW)':>10}"
    )
    print("-" * 120)
    bus_id_to_bus = {b.bus_i: b for b in mpc.bus}
    for br, flow in list(branches.items())[:top_n]:
        br_idx = int(br.split("_")[1])
        f_bus = mpc.branch[br_idx].f_bus
        t_bus = mpc.branch[br_idx].t_bus
        r = mpc.branch[br_idx].br_r
        x = mpc.branch[br_idx].br_x
        ratea = mpc.branch[br_idx].rate_a

        print(
            f"{f_bus:10} | {t_bus:10} | {r:10.4f} | {x:10.6f} | {flow:>16.4f} | {bus_id_to_bus[f_bus].va:10.5f} | {bus_id_to_bus[t_bus].va:10.5f} | {ratea:10.4f}"
        )
    print("=" * 120 + "\n")


def validate_matpower_energy_balance(
    case: Any,
    p_tol: float = 1e-3,
    q_tol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[bool, ValidationReport, Dict[Union[int, str], BusEnergyBalance]]:
    """Validate active and reactive power balance across all buses in a MATPOWER case.

    Args:
        case: MatpowerCase dataclass, MATPOWERCase dict, standard case dict, or RAW/m path.
        p_tol: Active power mismatch tolerance in MW (default: 0.001 MW).
        q_tol: Reactive power mismatch tolerance in MVAr (default: 0.001 MVAr).
        verbose: If True, print summary report and top violations table.

    Returns:
        Tuple of:
            - is_balanced (bool): True if all buses satisfy tolerance.
            - report (ValidationReport): Full validation report metrics.
            - id_to_bus (Dict[int, BusEnergyBalance]): Detailed per-bus balance records.
    """
    id_to_bus = build_bus_energy_balances(case)

    # Compute mismatches and totals
    p_mismatches: List[float] = []
    q_mismatches: List[float] = []
    max_p_val = 0.0
    max_q_val = 0.0
    max_p_bus: Union[int, str] = 0
    max_q_bus: Union[int, str] = 0

    total_p_gen = 0.0
    total_q_gen = 0.0
    total_p_load = 0.0
    total_q_load = 0.0
    total_p_shunt = 0.0
    total_q_shunt = 0.0

    for b_id, b in id_to_bus.items():
        if b.bus_type == 4:
            continue
        bal = calc_bus_balance(b)
        abs_p = abs(bal.p)
        abs_q = abs(bal.q)

        p_mismatches.append(bal.p)
        q_mismatches.append(bal.q)

        if abs_p > max_p_val:
            max_p_val = abs_p
            max_p_bus = b_id
        if abs_q > max_q_val:
            max_q_val = abs_q
            max_q_bus = b_id

        gen = b.total_gen()
        load = b.total_load()
        shunt = b.total_shunt()

        total_p_gen += gen.p
        total_q_gen += gen.q
        total_p_load += load.p
        total_q_load += load.q
        total_p_shunt += shunt.p
        total_q_shunt += shunt.q

    # RMS mismatch
    rms_p = math.sqrt(sum(p**2 for p in p_mismatches) / max(len(p_mismatches), 1))
    rms_q = math.sqrt(sum(q**2 for q in q_mismatches) / max(len(q_mismatches), 1))

    # Total branch losses = Gen - Load + Shunt
    total_p_loss = total_p_gen - total_p_load + total_p_shunt
    total_q_loss = total_q_gen - total_q_load + total_q_shunt

    violations = find_bus_balance_violations(
        id_to_bus, p_tol=p_tol, q_tol=q_tol, top_n=None, sort_by="p"
    )
    is_balanced = len(violations) == 0

    _, bus_data, gen_data, branch_data = _get_case_components(case)
    num_buses = len(id_to_bus)
    num_gens = (
        len(gen_data) if not isinstance(gen_data, np.ndarray) else gen_data.shape[0]
    )
    num_branches = (
        len(branch_data)
        if not isinstance(branch_data, np.ndarray)
        else branch_data.shape[0]
    )

    report = ValidationReport(
        is_balanced=is_balanced,
        num_buses=num_buses,
        num_generators=num_gens,
        num_branches=num_branches,
        max_p_mismatch=max_p_val,
        max_q_mismatch=max_q_val,
        rms_p_mismatch=rms_p,
        rms_q_mismatch=rms_q,
        max_p_bus=max_p_bus,
        max_q_bus=max_q_bus,
        total_p_gen=total_p_gen,
        total_q_gen=total_q_gen,
        total_p_load=total_p_load,
        total_q_load=total_q_load,
        total_p_shunt=total_p_shunt,
        total_q_shunt=total_q_shunt,
        total_p_loss=total_p_loss,
        total_q_loss=total_q_loss,
        tolerance_p=p_tol,
        tolerance_q=q_tol,
        num_violations=len(violations),
        violations=violations[:20],
    )

    if verbose:
        print(report.summary())
        if violations:
            print_bus_balance_summary(id_to_bus, top_n=10, p_tol=p_tol, q_tol=q_tol)

    return is_balanced, report, id_to_bus


def set_gen_bus_to_pv(mpc: MatpowerCase) -> MatpowerCase:
    gen_buses = {gen.gen_bus for gen in mpc.gen}
    for bus in mpc.bus:
        if bus.bus_i in gen_buses and bus.bus_type == 1:
            bus.bus_type = 2
            print(f"Set generator bus {bus.bus_i} to PV bus type from PQ bus type.")

    return mpc


def find_starbus_low_x(mpc: MatpowerCase) -> Dict[int, int]:
    g = nx.Graph()
    for br in mpc.branch:
        if abs(br.br_x) > 0.001:
            continue
        if br.f_bus not in mpc.star_buses and br.t_bus not in mpc.star_buses:
            continue
        g.add_edge(br.f_bus, br.t_bus, weight=br.br_x)

    cc = list(nx.connected_components(g))

    bus_del_to_keep = {}
    for island in cc:
        bus_to_keep = None
        for bus in island:
            if bus not in mpc.star_buses:
                bus_to_keep = bus
                break
        assert bus_to_keep is not None
        for bus in island:
            if bus != bus_to_keep:
                bus_del_to_keep[bus] = bus_to_keep

    return bus_del_to_keep


def is_buses_short_circuited(bus1: BusEnergyBalance, bus2: BusEnergyBalance):
    if abs(bus1.va - bus2.va) < 0.1:  # abs(bus1.vm - bus2.vm) < 0.01 and
        return True

    return False


def find_low_x_branches(
    mpc: MatpowerCase,
    id_to_bus: Dict[int, BusEnergyBalance],
    x_threshold: float = 0.001,
) -> Dict[int, int]:
    g = nx.Graph()
    for br in mpc.branch:
        if abs(br.br_x) > x_threshold:
            continue
        if not is_buses_short_circuited(id_to_bus[br.f_bus], id_to_bus[br.t_bus]):
            continue
        g.add_edge(br.f_bus, br.t_bus)

    cc = list(nx.connected_components(g))

    bus_del_to_keep = {}
    for island in cc:
        bus_to_keep = None
        for bus in island:
            if bus not in mpc.star_buses:
                bus_to_keep = bus
                break
        assert bus_to_keep is not None
        for bus in island:
            if bus != bus_to_keep:
                bus_del_to_keep[bus] = bus_to_keep

    return bus_del_to_keep


def open_branch_connect_to_del_buses(
    mpc: MatpowerCase, bus_del_to_keep: Dict[int, int]
) -> MatpowerCase:
    for br in mpc.branch:
        if br.f_bus in bus_del_to_keep and br.t_bus == bus_del_to_keep[br.f_bus]:
            br.br_status = 0
            print(f"Open branch between {br.f_bus} and {br.t_bus} with x = {br.br_x}")

        if br.t_bus in bus_del_to_keep and br.f_bus == bus_del_to_keep[br.t_bus]:
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

    for bus in mpc.bus:
        if bus.bus_i in bus_del_to_keep:
            bus.bus_type = 4
            print(f"Set bus id {bus.bus_i} as isolated bus.")

    return mpc


# =============================================================================
# Execution and Demonstration
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    from psse_raw import parse_raw
    from matpow import MatpowerCase

    # raw_file = (
    #     "/usr/local/google/home/sxzhou/Downloads/2025 Series RTEP 2030 SUM_06182025.raw"
    # )

    # print(f"Loading and converting RAW file: {raw_file}")
    # raw_data = parse_raw(raw_file)
    # mpc = MatpowerCase.from_psse(raw_data)

    mpc = MatpowerCase.from_mat(
        "/usr/local/google/home/sxzhou/Downloads/main_island.mat"
    )

    mpc = set_gen_bus_to_pv(mpc)
    id_to_bus = build_bus_energy_balances(mpc)
    bus_del_to_keep = find_low_x_branches(mpc, id_to_bus, x_threshold=0.001)
    mpc = open_branch_connect_to_del_buses(mpc, bus_del_to_keep)
    mpc = replace_bus_ids(mpc, bus_del_to_keep)

    # bus_del_to_keep = find_starbus_low_x(mpc)
    # mpc = open_branch_connect_to_del_buses(mpc, bus_del_to_keep)
    # mpc = replace_bus_ids(mpc, bus_del_to_keep)

    # print("\nPre-filtering low-impedance branches and short-circuit buses...")
    # disconnected_sc, sc_components = find_short_circuit_buses(mpc, id_to_bus)
    # bus_mappings = get_equivalent_buses(sc_components, star_buses)

    # print(f"  Short-circuit bus groups:            {len(sc_components)}")

    # Re-evaluate energy balance after topology updates
    id_to_bus = build_bus_energy_balances(mpc)
    balanced, rep, _ = validate_matpower_energy_balance(
        mpc, p_tol=1.0, q_tol=1.0, verbose=True
    )
    print_branch_large_flows(id_to_bus, mpc, top_n=10)

    # Verify dictionary export with numpy matrices
    d = mpc.to_dict()
    print("\nMATPOWER dictionary matrix shapes:")
    print(f"  bus matrix:    {d['bus'].shape}")
    print(f"  gen matrix:    {d['gen'].shape}")
    print(f"  branch matrix: {d['branch'].shape}")

    from pypower.api import runpf
    from pypower.ppoption import ppoption

    ppc = mpc.to_dict()

    ppopt = ppoption(
        MODEL="AC",  # AC power flow model
        PF_ALG=1,  # 1 = Newton-Raphson ('NR')
        PF_TOL=1e-8,  # Convergence tolerance
        PF_MAX_IT=1,  # Maximum iteration limit
        ENFORCE_Q_LIMS=0,  # 0 = Do not enforce Q limits initially
        VERBOSE=0,  # 2 = Print detailed progress
    )
    results, success = runpf(ppc, ppopt)
    print(f"\nPower flow converged: {bool(success)}")
