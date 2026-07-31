"""PSS/E RAW file parser and data structures.

Supports PSS/E revision 30, 32, 33, 34, and 35 RAW format power flow cases.
Provides object-oriented dataclasses, fast parsing, indexing, star bus
extraction (for 3-winding transformers), network summary statistics,
and optional pandas DataFrame export.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# =============================================================================
# Helper Parsing Functions
# =============================================================================


def _safe_int(v: Any, default: int = 0) -> int:
    """Safely convert value to int with fallback default."""
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback default."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_str(v: Any, default: str = "") -> str:
    """Safely strip quotes and whitespace from string."""
    if v is None:
        return default
    s = str(v).strip()
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        return s[1:-1].strip()
    return s.strip("' ")


def _split_raw_line(line: str) -> List[str]:
    """Fast splitter for comma-delimited PSS/E lines with quote and inline comment support."""
    line = line.rstrip("\r\n")
    if not line:
        return []

    # Fast-path for lines without inline comment character '/'
    if "/" not in line:
        return [p.strip().strip("' ") for p in line.split(",")]

    items: List[str] = []
    cur: List[str] = []
    in_quote = False

    for c in line:
        if c == "'":
            in_quote = not in_quote
            cur.append(c)
        elif c == "/" and not in_quote:
            # Inline comment begins
            break
        elif c == "," and not in_quote:
            token = "".join(cur).strip().strip("' ")
            items.append(token)
            cur = []
        else:
            cur.append(c)

    if cur:
        token = "".join(cur).strip().strip("' ")
        items.append(token)

    return items


# =============================================================================
# Component Dataclasses
# =============================================================================


@dataclass
class CaseIdentification:
    """PSS/E RAW Case Identification Header."""

    ic: int = 0
    sbase: float = 100.0
    rev: int = 35
    xfrrat: int = 0
    nxfrat: int = 0
    basfrq: float = 60.0
    title1: str = ""
    title2: str = ""


@dataclass
class Bus:
    """PSS/E Bus Record."""

    i: int
    name: str
    baskv: float
    ide: int = 1  # 1=PQ (load), 2=PV (gen), 3=Slack/Swing, 4=Disconnected
    area: int = 1
    zone: int = 1
    owner: int = 1
    vm: float = 1.0  # Voltage magnitude (pu)
    va: float = 0.0  # Voltage angle (deg)
    nvhi: float = 1.1
    nvlo: float = 0.9
    evhi: float = 1.1
    evlo: float = 0.9

    @property
    def is_load(self) -> bool:
        return self.ide == 1

    @property
    def is_gen(self) -> bool:
        return self.ide == 2

    @property
    def is_slack(self) -> bool:
        return self.ide == 3

    @property
    def is_isolated(self) -> bool:
        return self.ide == 4

    @property
    def is_in_service(self) -> bool:
        return self.ide != 4


@dataclass
class Load:
    """PSS/E Load Record."""

    i: int
    id: str = "1"
    status: int = 1  # 1=in-service, 0=out-of-service
    area: int = 1
    zone: int = 1
    pl: float = 0.0  # Active power (MW)
    ql: float = 0.0  # Reactive power (MVAR)
    ip: float = 0.0  # Active current power (MW at 1.0 pu V)
    iq: float = 0.0  # Reactive current power (MVAR at 1.0 pu V)
    yp: float = 0.0  # Active admittance power (MW at 1.0 pu V)
    yq: float = 0.0  # Reactive admittance power (MVAR at 1.0 pu V)
    owner: int = 1
    scale: int = 1
    intrpt: int = 0
    dgenp: float = 0.0
    dgenq: float = 0.0
    dgenf: float = 0.0
    load_name: str = ""

    @property
    def is_in_service(self) -> bool:
        return self.status == 1

    @property
    def p_mw(self) -> float:
        return self.pl if self.is_in_service else 0.0

    @property
    def q_mvar(self) -> float:
        return self.ql if self.is_in_service else 0.0


@dataclass
class FixedShunt:
    """PSS/E Fixed Bus Shunt Record."""

    i: int
    id: str = "1"
    status: int = 1  # 1=in-service, 0=out-of-service
    gl: float = 0.0  # Active component (MW at 1.0 pu V)
    bl: float = 0.0  # Reactive component (MVAR at 1.0 pu V)

    @property
    def is_in_service(self) -> bool:
        return self.status == 1


@dataclass
class Generator:
    """PSS/E Generator Record."""

    i: int
    id: str = "1"
    pg: float = 0.0  # MW output
    qg: float = 0.0  # MVAR output
    qt: float = 9999.0  # Max MVAR
    qb: float = -9999.0  # Min MVAR
    vs: float = 1.0  # Regulated voltage setpoint (pu)
    ireg: int = 0  # Regulated bus number
    mbase: float = 100.0  # MVA base
    zr: float = 0.0  # Machine resistance (pu)
    zx: float = 1.0  # Machine reactance (pu)
    rt: float = 0.0  # Step-up transformer resistance (pu)
    xt: float = 0.0  # Step-up transformer reactance (pu)
    gtap: float = 1.0  # Step-up transformer tap
    stat: int = 1  # 1=in-service, 0=out-of-service
    rmpct: float = 100.0  # Percent MVAR participation
    pt: float = 9999.0  # Max MW
    pb: float = -9999.0  # Min MW
    owners: List[Tuple[int, float]] = field(default_factory=list)
    wmod: int = 0  # Wind machine control mode
    wpf: float = 1.0  # Wind power factor

    @property
    def is_in_service(self) -> bool:
        return self.stat == 1

    @property
    def p_mw(self) -> float:
        return self.pg if self.is_in_service else 0.0

    @property
    def q_mvar(self) -> float:
        return self.qg if self.is_in_service else 0.0


@dataclass
class Branch:
    """PSS/E AC Transmission Line Branch Record."""

    i: int
    j: int
    ckt: str = "1"
    r: float = 0.0  # Resistance (pu)
    x: float = 0.0001  # Reactance (pu)
    b: float = 0.0  # Charging susceptance (pu)
    ratea: float = 0.0  # MVA rating A
    rateb: float = 0.0  # MVA rating B
    ratec: float = 0.0  # MVA rating C
    rates: List[float] = field(default_factory=list)
    gi: float = 0.0  # Line shunt conductance at bus i (pu)
    bi: float = 0.0  # Line shunt susceptance at bus i (pu)
    gj: float = 0.0  # Line shunt conductance at bus j (pu)
    bj: float = 0.0  # Line shunt susceptance at bus j (pu)
    st: int = 1  # 1=in-service, 0=out-of-service
    met: int = 1  # Metered end: 1=from (i), 2=to (j)
    len: float = 0.0  # Line length
    owners: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def is_in_service(self) -> bool:
        return self.st == 1

    @property
    def branch_id(self) -> Tuple[int, int, str]:
        return (self.i, self.j, self.ckt)


@dataclass
class TransformerWinding:
    """PSS/E Transformer Winding Parameters."""

    windv: float = 1.0  # Off-nominal turns ratio (pu) or winding voltage (kV)
    nomv: float = 0.0  # Nominal winding voltage (kV)
    ang: float = 0.0  # Phase shift angle (deg)
    rata: float = 0.0  # Rating A (MVA)
    ratb: float = 0.0  # Rating B (MVA)
    ratc: float = 0.0  # Rating C (MVA)
    cod: int = 0  # Tap control mode
    cont: int = 0  # Controlled bus number
    rma: float = 1.1  # Max tap ratio
    rmi: float = 0.9  # Min tap ratio
    vma: float = 1.1  # Max controlled voltage
    vmi: float = 0.9  # Min controlled voltage
    ntp: int = 33  # Number of tap positions
    tab: int = 0  # Impedance correction table index
    cr: float = 0.0  # Load drop comp resistance
    cx: float = 0.0  # Load drop comp reactance
    cnax: float = 0.0


@dataclass
class Transformer2W:
    """PSS/E 2-Winding Transformer Record."""

    i: int
    j: int
    k: int = 0
    ckt: str = "1"
    cw: int = 1
    cz: int = 1
    cm: int = 1
    mag1: float = 0.0  # Magnetizing conductance (pu)
    mag2: float = 0.0  # Magnetizing susceptance (pu)
    nmetr: int = 2
    name: str = ""
    stat: int = 1  # 1=in-service, 0=out-of-service
    owners: List[Tuple[int, float]] = field(default_factory=list)
    vecgrp: str = ""
    r1_2: float = 0.0
    x1_2: float = 0.0001
    sbase1_2: float = 100.0
    wdg1: TransformerWinding = field(default_factory=TransformerWinding)
    wdg2: TransformerWinding = field(default_factory=TransformerWinding)

    @property
    def is_in_service(self) -> bool:
        return self.stat == 1

    @property
    def transformer_id(self) -> Tuple[int, int, str]:
        return (self.i, self.j, self.ckt)


@dataclass
class Transformer3W:
    """PSS/E 3-Winding Transformer Record."""

    i: int
    j: int
    k: int
    ckt: str = "1"
    cw: int = 1
    cz: int = 1
    cm: int = 1
    mag1: float = 0.0  # Magnetizing conductance (pu)
    mag2: float = 0.0  # Magnetizing susceptance (pu)
    nmetr: int = 2
    name: str = ""
    stat: int = 1  # 1=in-service, 0=out-of-service
    owners: List[Tuple[int, float]] = field(default_factory=list)
    vecgrp: str = ""
    r1_2: float = 0.0
    x1_2: float = 0.0001
    sbase1_2: float = 100.0
    r2_3: float = 0.0
    x2_3: float = 0.0001
    sbase2_3: float = 100.0
    r3_1: float = 0.0
    x3_1: float = 0.0001
    sbase3_1: float = 100.0
    vmstar: float = 1.0  # Star bus voltage magnitude (pu)
    anstar: float = 0.0  # Star bus voltage angle (deg)
    wdg1: TransformerWinding = field(default_factory=TransformerWinding)
    wdg2: TransformerWinding = field(default_factory=TransformerWinding)
    wdg3: TransformerWinding = field(default_factory=TransformerWinding)

    @property
    def is_in_service(self) -> bool:
        return self.stat == 1

    @property
    def star_bus_name(self) -> str:
        """Standard star bus naming convention: T-i-j-k-ckt."""
        return f"T-{self.i}-{self.j}-{self.k}-{self.ckt}"

    @property
    def transformer_id(self) -> Tuple[int, int, int, str]:
        return (self.i, self.j, self.k, self.ckt)


@dataclass
class Area:
    """PSS/E Area Record."""

    i: int
    isw: int = 0
    pdes: float = 0.0
    ptol: float = 10.0
    arname: str = ""


@dataclass
class TwoTerminalDc:
    """PSS/E Two-Terminal DC Transmission Line Record."""

    name: str
    mdc: int = 0
    rdc: float = 0.0
    setvl: float = 0.0
    vschd: float = 0.0
    vcmod: float = 0.0
    rcomp: float = 0.0
    delti: float = 0.0
    meter: str = "I"
    dcvmin: float = 0.0
    ccconv: int = 0
    varfrac: float = 1.0
    rectifier_tokens: List[str] = field(default_factory=list)
    inverter_tokens: List[str] = field(default_factory=list)


@dataclass
class VscDc:
    """PSS/E Voltage Source Converter (VSC) DC Line Record."""

    name: str
    nconv: int = 2
    rdc: float = 0.0
    mode: int = 1
    sbase: float = 100.0
    conv1_tokens: List[str] = field(default_factory=list)
    conv2_tokens: List[str] = field(default_factory=list)


@dataclass
class ImpedanceCorrection:
    """PSS/E Transformer Impedance Correction Table."""

    i: int
    points: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class MultiTerminalDc:
    """PSS/E Multi-Terminal DC Line Record."""

    name: str
    nconv: int = 0
    ndcbs: int = 0
    ndcln: int = 0
    mdc: int = 0
    vconv: int = 0
    vcmod: float = 0.0
    vconvn: float = 0.0
    converters: List[List[str]] = field(default_factory=list)
    dc_buses: List[List[str]] = field(default_factory=list)
    dc_links: List[List[str]] = field(default_factory=list)


@dataclass
class MultiSectionLine:
    """PSS/E Multi-Section Line Grouping Record."""

    i: int
    j: int
    id: str = "&1"
    met: int = 1
    dum_buses: List[int] = field(default_factory=list)


@dataclass
class Zone:
    """PSS/E Zone Record."""

    i: int
    zoname: str = ""


@dataclass
class InterAreaTransfer:
    """PSS/E Inter-Area Transfer Record."""

    arfrom: int
    arto: int
    trid: str = "1"
    pdr: float = 0.0


@dataclass
class Owner:
    """PSS/E Owner Record."""

    i: int
    owname: str = ""


@dataclass
class Facts:
    """PSS/E FACTS Control Device Record."""

    name: str
    i: int
    j: int = 0
    mode: int = 1
    pset: float = 0.0
    qset: float = 0.0
    vset: float = 1.0
    shmx: float = 9999.0
    trmx: float = 9999.0
    vtdc: float = 0.0
    tmax: float = 1.1
    tmin: float = 0.9
    tokens: List[str] = field(default_factory=list)


@dataclass
class SwitchedShuntBlock:
    """Switched Shunt Step Block."""

    n: int = 0  # Number of steps
    b: float = 0.0  # MVAR per step


@dataclass
class SwitchedShunt:
    """PSS/E Switched Shunt Record."""

    i: int
    id: str = "1"
    modsw: int = 1
    adjm: int = 0
    stat: int = 1  # 1=in-service, 0=out-of-service
    vswhi: float = 1.0
    vswlo: float = 1.0
    swrem: int = 0
    rmpct: float = 100.0
    rmident: str = ""
    binit: float = 0.0
    blocks: List[SwitchedShuntBlock] = field(default_factory=list)

    @property
    def is_in_service(self) -> bool:
        return self.stat == 1


@dataclass
class GneDevice:
    """PSS/E GNE Device Record."""

    name: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class InductionMachine:
    """PSS/E Induction Machine Record."""

    i: int
    id: str = "1"
    tokens: List[str] = field(default_factory=list)


@dataclass
class Substation:
    """PSS/E Substation Record."""

    i: int
    name: str = ""
    tokens: List[str] = field(default_factory=list)


# =============================================================================
# Main Data Container
# =============================================================================


@dataclass
class PsseRawData:
    """Container holding all parsed components of a PSS/E RAW power flow case."""

    case_id: CaseIdentification = field(default_factory=CaseIdentification)
    buses: Dict[int, Bus] = field(default_factory=dict)
    loads: List[Load] = field(default_factory=list)
    fixed_shunts: List[FixedShunt] = field(default_factory=list)
    generators: List[Generator] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    transformers_2w: List[Transformer2W] = field(default_factory=list)
    transformers_3w: List[Transformer3W] = field(default_factory=list)
    areas: Dict[int, Area] = field(default_factory=dict)
    two_terminal_dcs: List[TwoTerminalDc] = field(default_factory=list)
    vsc_dcs: List[VscDc] = field(default_factory=list)
    impedance_corrections: Dict[int, ImpedanceCorrection] = field(default_factory=dict)
    multi_terminal_dcs: List[MultiTerminalDc] = field(default_factory=list)
    multi_section_lines: List[MultiSectionLine] = field(default_factory=list)
    zones: Dict[int, Zone] = field(default_factory=dict)
    inter_area_transfers: List[InterAreaTransfer] = field(default_factory=list)
    owners: Dict[int, Owner] = field(default_factory=dict)
    facts: List[Facts] = field(default_factory=list)
    switched_shunts: List[SwitchedShunt] = field(default_factory=list)
    gne_devices: List[GneDevice] = field(default_factory=list)
    induction_machines: List[InductionMachine] = field(default_factory=list)
    substations: List[Substation] = field(default_factory=list)

    # Internal index caches
    _loads_by_bus: Optional[Dict[int, List[Load]]] = None
    _generators_by_bus: Optional[Dict[int, List[Generator]]] = None
    _branches_by_bus: Optional[Dict[int, List[Branch]]] = None

    def get_bus(self, bus_id: int) -> Optional[Bus]:
        """Look up bus by bus ID."""
        return self.buses.get(bus_id)

    def get_loads(self, bus_id: Optional[int] = None) -> List[Load]:
        """Get all loads, or loads connected to a specific bus."""
        if bus_id is None:
            return self.loads
        if self._loads_by_bus is None:
            self._build_indices()
        return self._loads_by_bus.get(bus_id, [])

    def get_generators(self, bus_id: Optional[int] = None) -> List[Generator]:
        """Get all generators, or generators connected to a specific bus."""
        if bus_id is None:
            return self.generators
        if self._generators_by_bus is None:
            self._build_indices()
        return self._generators_by_bus.get(bus_id, [])

    def get_branches(self, bus_id: Optional[int] = None) -> List[Branch]:
        """Get all branches, or branches incident to a specific bus."""
        if bus_id is None:
            return self.branches
        if self._branches_by_bus is None:
            self._build_indices()
        return self._branches_by_bus.get(bus_id, [])

    def get_star_buses(self) -> Dict[str, Dict[str, Any]]:
        """Extract star bus voltages and angles from all 3-winding transformers.

        Returns a dictionary mapping `star_bus_name` (e.g. 'T-100002-103027-103056-1')
        to dict containing {'vm': vmstar, 'va': anstar, 'i': bus_i, 'j': bus_j, 'k': bus_k, 'ckt': ckt}.
        """
        star_buses: Dict[str, Dict[str, Any]] = {}
        for t in self.transformers_3w:
            star_buses[t.star_bus_name] = {
                "vm": t.vmstar,
                "va": t.anstar,
                "i": t.i,
                "j": t.j,
                "k": t.k,
                "ckt": t.ckt,
                "name": t.name,
                "stat": t.stat,
            }
        return star_buses

    def _build_indices(self) -> None:
        """Build bus-indexed lookups."""
        loads_map: Dict[int, List[Load]] = {}
        for load in self.loads:
            loads_map.setdefault(load.i, []).append(load)
        self._loads_by_bus = loads_map

        gens_map: Dict[int, List[Generator]] = {}
        for gen in self.generators:
            gens_map.setdefault(gen.i, []).append(gen)
        self._generators_by_bus = gens_map

        branches_map: Dict[int, List[Branch]] = {}
        for br in self.branches:
            branches_map.setdefault(br.i, []).append(br)
            branches_map.setdefault(br.j, []).append(br)
        self._branches_by_bus = branches_map

    def summary(self) -> Dict[str, Any]:
        """Compute key summary metrics for the case."""
        total_p_gen = sum(g.p_mw for g in self.generators if g.is_in_service)
        total_q_gen = sum(g.q_mvar for g in self.generators if g.is_in_service)
        total_p_load = sum(l.p_mw for l in self.loads if l.is_in_service)
        total_q_load = sum(l.q_mvar for l in self.loads if l.is_in_service)

        return {
            "rev": self.case_id.rev,
            "sbase": self.case_id.sbase,
            "basfrq": self.case_id.basfrq,
            "title1": self.case_id.title1,
            "title2": self.case_id.title2,
            "buses_count": len(self.buses),
            "loads_count": len(self.loads),
            "fixed_shunts_count": len(self.fixed_shunts),
            "generators_count": len(self.generators),
            "branches_count": len(self.branches),
            "transformers_2w_count": len(self.transformers_2w),
            "transformers_3w_count": len(self.transformers_3w),
            "areas_count": len(self.areas),
            "zones_count": len(self.zones),
            "owners_count": len(self.owners),
            "two_terminal_dcs_count": len(self.two_terminal_dcs),
            "vsc_dcs_count": len(self.vsc_dcs),
            "switched_shunts_count": len(self.switched_shunts),
            "facts_count": len(self.facts),
            "total_gen_mw": round(total_p_gen, 2),
            "total_gen_mvar": round(total_q_gen, 2),
            "total_load_mw": round(total_p_load, 2),
            "total_load_mvar": round(total_q_load, 2),
        }

    # -------------------------------------------------------------------------
    # Optional Pandas DataFrame Converters
    # -------------------------------------------------------------------------
    def buses_df(self) -> Any:
        """Export buses to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(b) for b in self.buses.values()])

    def loads_df(self) -> Any:
        """Export loads to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(l) for l in self.loads])

    def generators_df(self) -> Any:
        """Export generators to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(g) for g in self.generators])

    def branches_df(self) -> Any:
        """Export branches to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(b) for b in self.branches])

    def transformers_2w_df(self) -> Any:
        """Export 2-winding transformers to pandas DataFrame."""
        import pandas as pd

        records = []
        for t in self.transformers_2w:
            d = vars(t).copy()
            d.pop("wdg1", None)
            d.pop("wdg2", None)
            d["windv1"] = t.wdg1.windv
            d["nomv1"] = t.wdg1.nomv
            d["ang1"] = t.wdg1.ang
            d["rata1"] = t.wdg1.rata
            d["windv2"] = t.wdg2.windv
            d["nomv2"] = t.wdg2.nomv
            records.append(d)
        return pd.DataFrame(records)

    def transformers_3w_df(self) -> Any:
        """Export 3-winding transformers to pandas DataFrame."""
        import pandas as pd

        records = []
        for t in self.transformers_3w:
            d = vars(t).copy()
            d.pop("wdg1", None)
            d.pop("wdg2", None)
            d.pop("wdg3", None)
            d["star_bus_name"] = t.star_bus_name
            d["windv1"] = t.wdg1.windv
            d["nomv1"] = t.wdg1.nomv
            d["windv2"] = t.wdg2.windv
            d["nomv2"] = t.wdg2.nomv
            d["windv3"] = t.wdg3.windv
            d["nomv3"] = t.wdg3.nomv
            records.append(d)
        return pd.DataFrame(records)

    def switched_shunts_df(self) -> Any:
        """Export switched shunts to pandas DataFrame."""
        import pandas as pd

        records = []
        for s in self.switched_shunts:
            d = vars(s).copy()
            d.pop("blocks", None)
            d["num_blocks"] = len(s.blocks)
            records.append(d)
        return pd.DataFrame(records)

    def areas_df(self) -> Any:
        """Export areas to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(a) for a in self.areas.values()])

    def zones_df(self) -> Any:
        """Export zones to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(z) for z in self.zones.values()])

    def owners_df(self) -> Any:
        """Export owners to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([vars(o) for o in self.owners.values()])

    def to_dataframes(self) -> Dict[str, Any]:
        """Export all major components as a dictionary of pandas DataFrames."""
        return {
            "buses": self.buses_df(),
            "loads": self.loads_df(),
            "generators": self.generators_df(),
            "branches": self.branches_df(),
            "transformers_2w": self.transformers_2w_df(),
            "transformers_3w": self.transformers_3w_df(),
            "switched_shunts": self.switched_shunts_df(),
            "areas": self.areas_df(),
            "zones": self.zones_df(),
            "owners": self.owners_df(),
        }

    def to_networkx(
        self, include_transformers: bool = True, only_in_service: bool = True
    ) -> Any:
        """Construct a networkx Graph from the power flow network.

        Nodes are bus numbers with attributes: name, baskv, vm, va, ide, area, zone.
        Edges represent AC branches and 2W/3W transformers with impedance/ratings.
        """
        import networkx as nx

        G = nx.Graph()
        for bus in self.buses.values():
            if only_in_service and not bus.is_in_service:
                continue
            G.add_node(
                bus.i,
                name=bus.name,
                baskv=bus.baskv,
                vm=bus.vm,
                va=bus.va,
                ide=bus.ide,
                area=bus.area,
                zone=bus.zone,
                owner=bus.owner,
            )

        for br in self.branches:
            if only_in_service and not br.is_in_service:
                continue
            G.add_edge(
                br.i,
                br.j,
                key=br.ckt,
                r=br.r,
                x=br.x,
                b=br.b,
                ratea=br.ratea,
                branch_type="branch",
                len=br.len,
            )

        if include_transformers:
            for t2 in self.transformers_2w:
                if only_in_service and not t2.is_in_service:
                    continue
                G.add_edge(
                    t2.i,
                    t2.j,
                    key=t2.ckt,
                    r=t2.r1_2,
                    x=t2.x1_2,
                    sbase=t2.sbase1_2,
                    name=t2.name,
                    branch_type="transformer_2w",
                )

            for t3 in self.transformers_3w:
                if only_in_service and not t3.is_in_service:
                    continue
                star_node = t3.star_bus_name
                G.add_node(
                    star_node,
                    name=star_node,
                    vm=t3.vmstar,
                    va=t3.anstar,
                    is_star_bus=True,
                )
                G.add_edge(
                    t3.i,
                    star_node,
                    r=t3.r1_2,
                    x=t3.x1_2,
                    branch_type="transformer_3w_leg1",
                )
                G.add_edge(
                    t3.j,
                    star_node,
                    r=t3.r2_3,
                    x=t3.x2_3,
                    branch_type="transformer_3w_leg2",
                )
                G.add_edge(
                    t3.k,
                    star_node,
                    r=t3.r3_1,
                    x=t3.x3_1,
                    branch_type="transformer_3w_leg3",
                )

        return G


# =============================================================================
# Parser Implementation
# =============================================================================


class PsseRawParser:
    """Parser for PSS/E RAW power flow format files."""

    SECTION_KEYWORDS = [
        ("BUS DATA", "BUS"),
        ("LOAD DATA", "LOAD"),
        ("FIXED SHUNT DATA", "FIXED_SHUNT"),
        ("GENERATOR DATA", "GENERATOR"),
        ("BRANCH DATA", "BRANCH"),
        ("SYSTEM SWITCHING DEVICE DATA", "SYSTEM_SWITCHING"),
        ("TRANSFORMER DATA", "TRANSFORMER"),
        ("AREA DATA", "AREA"),
        ("TWO-TERMINAL DC DATA", "TWO_TERMINAL_DC"),
        ("VOLTAGE SOURCE CONVERTER DATA", "VSC_DC"),
        ("IMPEDANCE CORRECTION DATA", "IMPEDANCE_CORRECTION"),
        ("MULTI-TERMINAL DC DATA", "MULTI_TERMINAL_DC"),
        ("MULTI-SECTION LINE DATA", "MULTI_SECTION_LINE"),
        ("ZONE DATA", "ZONE"),
        ("INTER-AREA TRANSFER DATA", "INTER_AREA_TRANSFER"),
        ("OWNER DATA", "OWNER"),
        ("FACTS CONTROL DEVICE DATA", "FACTS"),
        ("SWITCHED SHUNT DATA", "SWITCHED_SHUNT"),
        ("GNE DEVICE DATA", "GNE_DEVICE"),
        ("INDUCTION MACHINE DATA", "INDUCTION_MACHINE"),
        ("SUBSTATION DATA", "SUBSTATION"),
    ]

    SECTION_ORDER = [k[1] for k in SECTION_KEYWORDS]

    @classmethod
    def parse_file(cls, file_path: str) -> PsseRawData:
        """Parse a PSS/E RAW file given its filesystem path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RAW file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        return cls._parse_lines(lines)

    @classmethod
    def parse_string(cls, content: str) -> PsseRawData:
        """Parse PSS/E RAW content from a multiline string."""
        lines = content.splitlines(keepends=True)
        return cls._parse_lines(lines)

    @classmethod
    def _parse_lines(cls, lines: Sequence[str]) -> PsseRawData:
        """Internal line-by-line parsing engine."""
        data = PsseRawData()
        if not lines:
            return data

        # 1. Header (lines 0 to 2)
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        if idx < len(lines):
            header_toks = _split_raw_line(lines[idx])
            data.case_id = CaseIdentification(
                ic=_safe_int(header_toks[0] if len(header_toks) > 0 else 0),
                sbase=_safe_float(
                    header_toks[1] if len(header_toks) > 1 else 100.0, 100.0
                ),
                rev=_safe_int(header_toks[2] if len(header_toks) > 2 else 35, 35),
                xfrrat=_safe_int(header_toks[3] if len(header_toks) > 3 else 0),
                nxfrat=_safe_int(header_toks[4] if len(header_toks) > 4 else 0),
                basfrq=_safe_float(
                    header_toks[5] if len(header_toks) > 5 else 60.0, 60.0
                ),
            )
            idx += 1

        if idx < len(lines):
            data.case_id.title1 = lines[idx].strip()
            idx += 1

        if idx < len(lines):
            data.case_id.title2 = lines[idx].strip()
            idx += 1

        current_sec_idx = -1

        # 2. Main parsing loop
        while idx < len(lines):
            raw_line = lines[idx]
            line = raw_line.strip()
            if not line:
                idx += 1
                continue

            first_char = line[0]
            if first_char == "0" or line == "Q":
                parts = _split_raw_line(line)
                if line == "Q":
                    break
                is_delimiter = (
                    line == "0"
                    or line.startswith("0 ")
                    or line.startswith("0/")
                    or line.startswith("0 /")
                    or (len(parts) > 0 and parts[0] == "0")
                )

                if is_delimiter:
                    matched = False
                    upper = line.upper()
                    if "BEGIN" in upper:
                        for kw, sec_name in cls.SECTION_KEYWORDS:
                            if f"BEGIN {kw}" in upper or f"BEGIN OF {kw}" in upper:
                                current_sec_idx = cls.SECTION_ORDER.index(sec_name)
                                matched = True
                                break
                    elif "END OF" in upper:
                        for kw, sec_name in cls.SECTION_KEYWORDS:
                            if f"END OF {kw}" in upper:
                                current_sec_idx = cls.SECTION_ORDER.index(sec_name) + 1
                                matched = True
                                break
                    if not matched:
                        current_sec_idx += 1
                    idx += 1
                    continue

            if current_sec_idx < 0 or current_sec_idx >= len(cls.SECTION_ORDER):
                idx += 1
                continue

            sec = cls.SECTION_ORDER[current_sec_idx]
            toks = _split_raw_line(raw_line)
            if not toks or toks[0] == "0":
                idx += 1
                continue

            # Section-specific parsing
            if sec == "BUS":
                bus_i = _safe_int(toks[0])
                data.buses[bus_i] = Bus(
                    i=bus_i,
                    name=_safe_str(toks[1] if len(toks) > 1 else ""),
                    baskv=_safe_float(toks[2] if len(toks) > 2 else 0.0),
                    ide=_safe_int(toks[3] if len(toks) > 3 else 1, 1),
                    area=_safe_int(toks[4] if len(toks) > 4 else 1, 1),
                    zone=_safe_int(toks[5] if len(toks) > 5 else 1, 1),
                    owner=_safe_int(toks[6] if len(toks) > 6 else 1, 1),
                    vm=_safe_float(toks[7] if len(toks) > 7 else 1.0, 1.0),
                    va=_safe_float(toks[8] if len(toks) > 8 else 0.0, 0.0),
                    nvhi=_safe_float(toks[9] if len(toks) > 9 else 1.1, 1.1),
                    nvlo=_safe_float(toks[10] if len(toks) > 10 else 0.9, 0.9),
                    evhi=_safe_float(toks[11] if len(toks) > 11 else 1.1, 1.1),
                    evlo=_safe_float(toks[12] if len(toks) > 12 else 0.9, 0.9),
                )
                idx += 1

            elif sec == "LOAD":
                data.loads.append(
                    Load(
                        i=_safe_int(toks[0]),
                        id=_safe_str(toks[1] if len(toks) > 1 else "1", "1"),
                        status=_safe_int(toks[2] if len(toks) > 2 else 1, 1),
                        area=_safe_int(toks[3] if len(toks) > 3 else 1, 1),
                        zone=_safe_int(toks[4] if len(toks) > 4 else 1, 1),
                        pl=_safe_float(toks[5] if len(toks) > 5 else 0.0),
                        ql=_safe_float(toks[6] if len(toks) > 6 else 0.0),
                        ip=_safe_float(toks[7] if len(toks) > 7 else 0.0),
                        iq=_safe_float(toks[8] if len(toks) > 8 else 0.0),
                        yp=_safe_float(toks[9] if len(toks) > 9 else 0.0),
                        yq=_safe_float(toks[10] if len(toks) > 10 else 0.0),
                        owner=_safe_int(toks[11] if len(toks) > 11 else 1, 1),
                        scale=_safe_int(toks[12] if len(toks) > 12 else 1, 1),
                        intrpt=_safe_int(toks[13] if len(toks) > 13 else 0, 0),
                        dgenp=_safe_float(toks[14] if len(toks) > 14 else 0.0),
                        dgenq=_safe_float(toks[15] if len(toks) > 15 else 0.0),
                        dgenf=_safe_float(toks[16] if len(toks) > 16 else 0.0),
                        load_name=_safe_str(toks[17] if len(toks) > 17 else ""),
                    )
                )
                idx += 1

            elif sec == "FIXED_SHUNT":
                data.fixed_shunts.append(
                    FixedShunt(
                        i=_safe_int(toks[0]),
                        id=_safe_str(toks[1] if len(toks) > 1 else "1", "1"),
                        status=_safe_int(toks[2] if len(toks) > 2 else 1, 1),
                        gl=_safe_float(toks[3] if len(toks) > 3 else 0.0),
                        bl=_safe_float(toks[4] if len(toks) > 4 else 0.0),
                    )
                )
                idx += 1

            elif sec == "GENERATOR":
                owners: List[Tuple[int, float]] = []
                # Owners start at index 18 (O1, F1, O2, F2, O3, F3, O4, F4)
                for o_idx in range(18, min(26, len(toks)), 2):
                    o_num = _safe_int(toks[o_idx])
                    if o_num > 0:
                        o_frac = _safe_float(
                            toks[o_idx + 1] if o_idx + 1 < len(toks) else 1.0, 1.0
                        )
                        owners.append((o_num, o_frac))

                data.generators.append(
                    Generator(
                        i=_safe_int(toks[0]),
                        id=_safe_str(toks[1] if len(toks) > 1 else "1", "1"),
                        pg=_safe_float(toks[2] if len(toks) > 2 else 0.0),
                        qg=_safe_float(toks[3] if len(toks) > 3 else 0.0),
                        qt=_safe_float(toks[4] if len(toks) > 4 else 9999.0, 9999.0),
                        qb=_safe_float(toks[5] if len(toks) > 5 else -9999.0, -9999.0),
                        vs=_safe_float(toks[6] if len(toks) > 6 else 1.0, 1.0),
                        ireg=_safe_int(toks[7] if len(toks) > 7 else 0),
                        mbase=_safe_float(
                            toks[8] if len(toks) > 8 else data.case_id.sbase,
                            data.case_id.sbase,
                        ),
                        zr=_safe_float(toks[9] if len(toks) > 9 else 0.0),
                        zx=_safe_float(toks[10] if len(toks) > 10 else 1.0, 1.0),
                        rt=_safe_float(toks[11] if len(toks) > 11 else 0.0),
                        xt=_safe_float(toks[12] if len(toks) > 12 else 0.0),
                        gtap=_safe_float(toks[13] if len(toks) > 13 else 1.0, 1.0),
                        stat=_safe_int(toks[14] if len(toks) > 14 else 1, 1),
                        rmpct=_safe_float(toks[15] if len(toks) > 15 else 100.0, 100.0),
                        pt=_safe_float(toks[16] if len(toks) > 16 else 9999.0, 9999.0),
                        pb=_safe_float(
                            toks[17] if len(toks) > 17 else -9999.0, -9999.0
                        ),
                        owners=owners,
                        wmod=_safe_int(toks[26] if len(toks) > 26 else 0),
                        wpf=_safe_float(toks[27] if len(toks) > 27 else 1.0, 1.0),
                    )
                )
                idx += 1

            elif sec == "BRANCH":
                owners: List[Tuple[int, float]] = []
                # Owners in branch typically start at index 26
                for o_idx in range(26, min(34, len(toks)), 2):
                    o_num = _safe_int(toks[o_idx])
                    if o_num > 0:
                        o_frac = _safe_float(
                            toks[o_idx + 1] if o_idx + 1 < len(toks) else 1.0, 1.0
                        )
                        owners.append((o_num, o_frac))

                rates = [
                    _safe_float(toks[r_idx]) for r_idx in range(7, min(19, len(toks)))
                ]

                data.branches.append(
                    Branch(
                        i=_safe_int(toks[0]),
                        j=_safe_int(toks[1]),
                        ckt=_safe_str(toks[2] if len(toks) > 2 else "1", "1"),
                        r=_safe_float(toks[3] if len(toks) > 3 else 0.0),
                        x=_safe_float(toks[4] if len(toks) > 4 else 0.0001, 0.0001),
                        b=_safe_float(toks[5] if len(toks) > 5 else 0.0),
                        ratea=rates[0] if len(rates) > 0 else 0.0,
                        rateb=rates[1] if len(rates) > 1 else 0.0,
                        ratec=rates[2] if len(rates) > 2 else 0.0,
                        rates=rates,
                        gi=_safe_float(toks[19] if len(toks) > 19 else 0.0),
                        bi=_safe_float(toks[20] if len(toks) > 20 else 0.0),
                        gj=_safe_float(toks[21] if len(toks) > 21 else 0.0),
                        bj=_safe_float(toks[22] if len(toks) > 22 else 0.0),
                        st=_safe_int(toks[23] if len(toks) > 23 else 1, 1),
                        met=_safe_int(toks[24] if len(toks) > 24 else 1, 1),
                        len=_safe_float(toks[25] if len(toks) > 25 else 0.0),
                        owners=owners,
                    )
                )
                idx += 1

            elif sec == "TRANSFORMER":
                k = _safe_int(toks[2] if len(toks) > 2 else 0)
                l2 = _split_raw_line(lines[idx + 1]) if idx + 1 < len(lines) else []
                l3 = _split_raw_line(lines[idx + 2]) if idx + 2 < len(lines) else []
                l4 = _split_raw_line(lines[idx + 3]) if idx + 3 < len(lines) else []

                # Owners in transformer line 1
                owners: List[Tuple[int, float]] = []
                for o_idx in range(12, min(20, len(toks)), 2):
                    o_num = _safe_int(toks[o_idx])
                    if o_num > 0:
                        o_frac = _safe_float(
                            toks[o_idx + 1] if o_idx + 1 < len(toks) else 1.0, 1.0
                        )
                        owners.append((o_num, o_frac))

                wdg1 = TransformerWinding(
                    windv=_safe_float(l3[0] if len(l3) > 0 else 1.0, 1.0),
                    nomv=_safe_float(l3[1] if len(l3) > 1 else 0.0),
                    ang=_safe_float(l3[2] if len(l3) > 2 else 0.0),
                    rata=_safe_float(l3[3] if len(l3) > 3 else 0.0),
                    ratb=_safe_float(l3[4] if len(l3) > 4 else 0.0),
                    ratc=_safe_float(l3[5] if len(l3) > 5 else 0.0),
                    cod=_safe_int(l3[6] if len(l3) > 6 else 0),
                    cont=_safe_int(l3[7] if len(l3) > 7 else 0),
                    rma=_safe_float(l3[8] if len(l3) > 8 else 1.1, 1.1),
                    rmi=_safe_float(l3[9] if len(l3) > 9 else 0.9, 0.9),
                    vma=_safe_float(l3[10] if len(l3) > 10 else 1.1, 1.1),
                    vmi=_safe_float(l3[11] if len(l3) > 11 else 0.9, 0.9),
                    ntp=_safe_int(l3[12] if len(l3) > 12 else 33, 33),
                    tab=_safe_int(l3[13] if len(l3) > 13 else 0),
                    cr=_safe_float(l3[14] if len(l3) > 14 else 0.0),
                    cx=_safe_float(l3[15] if len(l3) > 15 else 0.0),
                    cnax=_safe_float(l3[16] if len(l3) > 16 else 0.0),
                )

                wdg2 = TransformerWinding(
                    windv=_safe_float(l4[0] if len(l4) > 0 else 1.0, 1.0),
                    nomv=_safe_float(l4[1] if len(l4) > 1 else 0.0),
                    ang=_safe_float(l4[2] if len(l4) > 2 else 0.0),
                    rata=_safe_float(l4[3] if len(l4) > 3 else 0.0),
                    ratb=_safe_float(l4[4] if len(l4) > 4 else 0.0),
                    ratc=_safe_float(l4[5] if len(l4) > 5 else 0.0),
                    cod=_safe_int(l4[6] if len(l4) > 6 else 0),
                    cont=_safe_int(l4[7] if len(l4) > 7 else 0),
                    rma=_safe_float(l4[8] if len(l4) > 8 else 1.1, 1.1),
                    rmi=_safe_float(l4[9] if len(l4) > 9 else 0.9, 0.9),
                    vma=_safe_float(l4[10] if len(l4) > 10 else 1.1, 1.1),
                    vmi=_safe_float(l4[11] if len(l4) > 11 else 0.9, 0.9),
                    ntp=_safe_int(l4[12] if len(l4) > 12 else 33, 33),
                    tab=_safe_int(l4[13] if len(l4) > 13 else 0),
                    cr=_safe_float(l4[14] if len(l4) > 14 else 0.0),
                    cx=_safe_float(l4[15] if len(l4) > 15 else 0.0),
                    cnax=_safe_float(l4[16] if len(l4) > 16 else 0.0),
                )

                if k == 0:
                    # 2-winding transformer (4 lines total)
                    data.transformers_2w.append(
                        Transformer2W(
                            i=_safe_int(toks[0]),
                            j=_safe_int(toks[1]),
                            k=0,
                            ckt=_safe_str(toks[3] if len(toks) > 3 else "1", "1"),
                            cw=_safe_int(toks[4] if len(toks) > 4 else 1, 1),
                            cz=_safe_int(toks[5] if len(toks) > 5 else 1, 1),
                            cm=_safe_int(toks[6] if len(toks) > 6 else 1, 1),
                            mag1=_safe_float(toks[7] if len(toks) > 7 else 0.0),
                            mag2=_safe_float(toks[8] if len(toks) > 8 else 0.0),
                            nmetr=_safe_int(toks[9] if len(toks) > 9 else 2, 2),
                            name=_safe_str(toks[10] if len(toks) > 10 else ""),
                            stat=_safe_int(toks[11] if len(toks) > 11 else 1, 1),
                            owners=owners,
                            vecgrp=_safe_str(toks[20] if len(toks) > 20 else ""),
                            r1_2=_safe_float(l2[0] if len(l2) > 0 else 0.0),
                            x1_2=_safe_float(l2[1] if len(l2) > 1 else 0.0001, 0.0001),
                            sbase1_2=_safe_float(
                                l2[2] if len(l2) > 2 else data.case_id.sbase,
                                data.case_id.sbase,
                            ),
                            wdg1=wdg1,
                            wdg2=wdg2,
                        )
                    )
                    idx += 4
                else:
                    # 3-winding transformer (5 lines total)
                    l5 = _split_raw_line(lines[idx + 4]) if idx + 4 < len(lines) else []
                    wdg3 = TransformerWinding(
                        windv=_safe_float(l5[0] if len(l5) > 0 else 1.0, 1.0),
                        nomv=_safe_float(l5[1] if len(l5) > 1 else 0.0),
                        ang=_safe_float(l5[2] if len(l5) > 2 else 0.0),
                        rata=_safe_float(l5[3] if len(l5) > 3 else 0.0),
                        ratb=_safe_float(l5[4] if len(l5) > 4 else 0.0),
                        ratc=_safe_float(l5[5] if len(l5) > 5 else 0.0),
                        cod=_safe_int(l5[6] if len(l5) > 6 else 0),
                        cont=_safe_int(l5[7] if len(l5) > 7 else 0),
                        rma=_safe_float(l5[8] if len(l5) > 8 else 1.1, 1.1),
                        rmi=_safe_float(l5[9] if len(l5) > 9 else 0.9, 0.9),
                        vma=_safe_float(l5[10] if len(l5) > 10 else 1.1, 1.1),
                        vmi=_safe_float(l5[11] if len(l5) > 11 else 0.9, 0.9),
                        ntp=_safe_int(l5[12] if len(l5) > 12 else 33, 33),
                        tab=_safe_int(l5[13] if len(l5) > 13 else 0),
                        cr=_safe_float(l5[14] if len(l5) > 14 else 0.0),
                        cx=_safe_float(l5[15] if len(l5) > 15 else 0.0),
                        cnax=_safe_float(l5[16] if len(l5) > 16 else 0.0),
                    )

                    data.transformers_3w.append(
                        Transformer3W(
                            i=_safe_int(toks[0]),
                            j=_safe_int(toks[1]),
                            k=k,
                            ckt=_safe_str(toks[3] if len(toks) > 3 else "1", "1"),
                            cw=_safe_int(toks[4] if len(toks) > 4 else 1, 1),
                            cz=_safe_int(toks[5] if len(toks) > 5 else 1, 1),
                            cm=_safe_int(toks[6] if len(toks) > 6 else 1, 1),
                            mag1=_safe_float(toks[7] if len(toks) > 7 else 0.0),
                            mag2=_safe_float(toks[8] if len(toks) > 8 else 0.0),
                            nmetr=_safe_int(toks[9] if len(toks) > 9 else 2, 2),
                            name=_safe_str(toks[10] if len(toks) > 10 else ""),
                            stat=_safe_int(toks[11] if len(toks) > 11 else 1, 1),
                            owners=owners,
                            vecgrp=_safe_str(toks[20] if len(toks) > 20 else ""),
                            r1_2=_safe_float(l2[0] if len(l2) > 0 else 0.0),
                            x1_2=_safe_float(l2[1] if len(l2) > 1 else 0.0001, 0.0001),
                            sbase1_2=_safe_float(
                                l2[2] if len(l2) > 2 else data.case_id.sbase,
                                data.case_id.sbase,
                            ),
                            r2_3=_safe_float(l2[3] if len(l2) > 3 else 0.0),
                            x2_3=_safe_float(l2[4] if len(l2) > 4 else 0.0001, 0.0001),
                            sbase2_3=_safe_float(
                                l2[5] if len(l2) > 5 else data.case_id.sbase,
                                data.case_id.sbase,
                            ),
                            r3_1=_safe_float(l2[6] if len(l2) > 6 else 0.0),
                            x3_1=_safe_float(l2[7] if len(l2) > 7 else 0.0001, 0.0001),
                            sbase3_1=_safe_float(
                                l2[8] if len(l2) > 8 else data.case_id.sbase,
                                data.case_id.sbase,
                            ),
                            vmstar=_safe_float(l2[9] if len(l2) > 9 else 1.0, 1.0),
                            anstar=_safe_float(l2[10] if len(l2) > 10 else 0.0),
                            wdg1=wdg1,
                            wdg2=wdg2,
                            wdg3=wdg3,
                        )
                    )
                    idx += 5

            elif sec == "AREA":
                area_i = _safe_int(toks[0])
                data.areas[area_i] = Area(
                    i=area_i,
                    isw=_safe_int(toks[1] if len(toks) > 1 else 0),
                    pdes=_safe_float(toks[2] if len(toks) > 2 else 0.0),
                    ptol=_safe_float(toks[3] if len(toks) > 3 else 10.0, 10.0),
                    arname=_safe_str(toks[4] if len(toks) > 4 else ""),
                )
                idx += 1

            elif sec == "TWO_TERMINAL_DC":
                l2 = _split_raw_line(lines[idx + 1]) if idx + 1 < len(lines) else []
                l3 = _split_raw_line(lines[idx + 2]) if idx + 2 < len(lines) else []
                data.two_terminal_dcs.append(
                    TwoTerminalDc(
                        name=_safe_str(toks[0]),
                        mdc=_safe_int(toks[1] if len(toks) > 1 else 0),
                        rdc=_safe_float(toks[2] if len(toks) > 2 else 0.0),
                        setvl=_safe_float(toks[3] if len(toks) > 3 else 0.0),
                        vschd=_safe_float(toks[4] if len(toks) > 4 else 0.0),
                        vcmod=_safe_float(toks[5] if len(toks) > 5 else 0.0),
                        rcomp=_safe_float(toks[6] if len(toks) > 6 else 0.0),
                        delti=_safe_float(toks[7] if len(toks) > 7 else 0.0),
                        meter=_safe_str(toks[8] if len(toks) > 8 else "I", "I"),
                        dcvmin=_safe_float(toks[9] if len(toks) > 9 else 0.0),
                        ccconv=_safe_int(toks[10] if len(toks) > 10 else 0),
                        varfrac=_safe_float(toks[11] if len(toks) > 11 else 1.0, 1.0),
                        rectifier_tokens=l2,
                        inverter_tokens=l3,
                    )
                )
                idx += 3

            elif sec == "VSC_DC":
                l2 = _split_raw_line(lines[idx + 1]) if idx + 1 < len(lines) else []
                l3 = _split_raw_line(lines[idx + 2]) if idx + 2 < len(lines) else []
                data.vsc_dcs.append(
                    VscDc(
                        name=_safe_str(toks[0]),
                        nconv=_safe_int(toks[1] if len(toks) > 1 else 2, 2),
                        rdc=_safe_float(toks[2] if len(toks) > 2 else 0.0),
                        mode=_safe_int(toks[3] if len(toks) > 3 else 1, 1),
                        sbase=_safe_float(
                            toks[4] if len(toks) > 4 else data.case_id.sbase,
                            data.case_id.sbase,
                        ),
                        conv1_tokens=l2,
                        conv2_tokens=l3,
                    )
                )
                idx += 3

            elif sec == "IMPEDANCE_CORRECTION":
                t_idx = _safe_int(toks[0])
                pts: List[Tuple[float, float, float]] = []
                all_tokens = toks[1:]
                if idx + 1 < len(lines) and not lines[idx + 1].strip().startswith("0"):
                    l2 = _split_raw_line(lines[idx + 1])
                    all_tokens.extend(l2)
                    idx += 2
                else:
                    idx += 1

                for p_i in range(0, len(all_tokens), 3):
                    if p_i + 1 < len(all_tokens):
                        t_val = _safe_float(all_tokens[p_i])
                        f_val = _safe_float(all_tokens[p_i + 1])
                        g_val = (
                            _safe_float(all_tokens[p_i + 2])
                            if p_i + 2 < len(all_tokens)
                            else 0.0
                        )
                        if t_val != 0.0 or f_val != 0.0 or g_val != 0.0:
                            pts.append((t_val, f_val, g_val))

                data.impedance_corrections[t_idx] = ImpedanceCorrection(
                    i=t_idx, points=pts
                )

            elif sec == "MULTI_TERMINAL_DC":
                name = _safe_str(toks[0])
                nconv = _safe_int(toks[1] if len(toks) > 1 else 0)
                ndcbs = _safe_int(toks[2] if len(toks) > 2 else 0)
                ndcln = _safe_int(toks[3] if len(toks) > 3 else 0)
                mdc = _safe_int(toks[4] if len(toks) > 4 else 0)
                vconv = _safe_int(toks[5] if len(toks) > 5 else 0)
                vcmod = _safe_float(toks[6] if len(toks) > 6 else 0.0)
                vconvn = _safe_float(toks[7] if len(toks) > 7 else 0.0)

                converters = [_split_raw_line(lines[idx + 1 + j]) for j in range(nconv)]
                dc_buses = [
                    _split_raw_line(lines[idx + 1 + nconv + j]) for j in range(ndcbs)
                ]
                dc_links = [
                    _split_raw_line(lines[idx + 1 + nconv + ndcbs + j])
                    for j in range(ndcln)
                ]

                data.multi_terminal_dcs.append(
                    MultiTerminalDc(
                        name=name,
                        nconv=nconv,
                        ndcbs=ndcbs,
                        ndcln=ndcln,
                        mdc=mdc,
                        vconv=vconv,
                        vcmod=vcmod,
                        vconvn=vconvn,
                        converters=converters,
                        dc_buses=dc_buses,
                        dc_links=dc_links,
                    )
                )
                idx += 1 + nconv + ndcbs + ndcln

            elif sec == "MULTI_SECTION_LINE":
                dum_buses = [_safe_int(tok) for tok in toks[4:] if _safe_int(tok) != 0]
                data.multi_section_lines.append(
                    MultiSectionLine(
                        i=_safe_int(toks[0]),
                        j=_safe_int(toks[1]),
                        id=_safe_str(toks[2] if len(toks) > 2 else "&1", "&1"),
                        met=_safe_int(toks[3] if len(toks) > 3 else 1, 1),
                        dum_buses=dum_buses,
                    )
                )
                idx += 1

            elif sec == "ZONE":
                zone_i = _safe_int(toks[0])
                data.zones[zone_i] = Zone(
                    i=zone_i,
                    zoname=_safe_str(toks[1] if len(toks) > 1 else ""),
                )
                idx += 1

            elif sec == "INTER_AREA_TRANSFER":
                data.inter_area_transfers.append(
                    InterAreaTransfer(
                        arfrom=_safe_int(toks[0]),
                        arto=_safe_int(toks[1]),
                        trid=_safe_str(toks[2] if len(toks) > 2 else "1", "1"),
                        pdr=_safe_float(toks[3] if len(toks) > 3 else 0.0),
                    )
                )
                idx += 1

            elif sec == "OWNER":
                owner_i = _safe_int(toks[0])
                data.owners[owner_i] = Owner(
                    i=owner_i,
                    owname=_safe_str(toks[1] if len(toks) > 1 else ""),
                )
                idx += 1

            elif sec == "FACTS":
                data.facts.append(
                    Facts(
                        name=_safe_str(toks[0]),
                        i=_safe_int(toks[1] if len(toks) > 1 else 0),
                        j=_safe_int(toks[2] if len(toks) > 2 else 0),
                        mode=_safe_int(toks[3] if len(toks) > 3 else 1, 1),
                        pset=_safe_float(toks[4] if len(toks) > 4 else 0.0),
                        qset=_safe_float(toks[5] if len(toks) > 5 else 0.0),
                        vset=_safe_float(toks[6] if len(toks) > 6 else 1.0, 1.0),
                        shmx=_safe_float(toks[7] if len(toks) > 7 else 9999.0, 9999.0),
                        trmx=_safe_float(toks[8] if len(toks) > 8 else 9999.0, 9999.0),
                        vtdc=_safe_float(toks[9] if len(toks) > 9 else 0.0),
                        tmax=_safe_float(toks[10] if len(toks) > 10 else 1.1, 1.1),
                        tmin=_safe_float(toks[11] if len(toks) > 11 else 0.9, 0.9),
                        tokens=toks,
                    )
                )
                idx += 1

            elif sec == "SWITCHED_SHUNT":
                # In PSS/E v33/35:
                # I, ID, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, RMIDENT, BINIT, N1, B1, N2, B2...
                # In PSS/E v30/32:
                # I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, RMIDENT, BINIT, N1, B1, ...
                shunt_i = _safe_int(toks[0])
                has_id = len(toks) > 1 and not toks[1].isdigit() and len(toks[1]) <= 2

                if has_id or len(toks) >= 12:
                    shunt_id = _safe_str(toks[1], "1")
                    modsw = _safe_int(toks[2] if len(toks) > 2 else 1, 1)
                    adjm = _safe_int(toks[3] if len(toks) > 3 else 0)
                    stat = _safe_int(toks[4] if len(toks) > 4 else 1, 1)
                    vswhi = _safe_float(toks[5] if len(toks) > 5 else 1.0, 1.0)
                    vswlo = _safe_float(toks[6] if len(toks) > 6 else 1.0, 1.0)
                    swrem = _safe_int(toks[7] if len(toks) > 7 else 0)
                    rmpct = _safe_float(toks[8] if len(toks) > 8 else 100.0, 100.0)
                    rmident = _safe_str(toks[9] if len(toks) > 9 else "")
                    binit = _safe_float(
                        toks[11]
                        if len(toks) > 11
                        else (toks[10] if len(toks) > 10 else 0.0)
                    )
                    block_start = 12 if len(toks) > 11 else 11
                else:
                    shunt_id = "1"
                    modsw = _safe_int(toks[1] if len(toks) > 1 else 1, 1)
                    adjm = _safe_int(toks[2] if len(toks) > 2 else 0)
                    stat = _safe_int(toks[3] if len(toks) > 3 else 1, 1)
                    vswhi = _safe_float(toks[4] if len(toks) > 4 else 1.0, 1.0)
                    vswlo = _safe_float(toks[5] if len(toks) > 5 else 1.0, 1.0)
                    swrem = _safe_int(toks[6] if len(toks) > 6 else 0)
                    rmpct = _safe_float(toks[7] if len(toks) > 7 else 100.0, 100.0)
                    rmident = _safe_str(toks[8] if len(toks) > 8 else "")
                    binit = _safe_float(toks[9] if len(toks) > 9 else 0.0)
                    block_start = 10

                blocks: List[SwitchedShuntBlock] = []
                for b_i in range(block_start, len(toks), 2):
                    n_steps = _safe_int(toks[b_i])
                    if n_steps != 0 and b_i + 1 < len(toks):
                        b_val = _safe_float(toks[b_i + 1])
                        blocks.append(SwitchedShuntBlock(n=n_steps, b=b_val))

                data.switched_shunts.append(
                    SwitchedShunt(
                        i=shunt_i,
                        id=shunt_id,
                        modsw=modsw,
                        adjm=adjm,
                        stat=stat,
                        vswhi=vswhi,
                        vswlo=vswlo,
                        swrem=swrem,
                        rmpct=rmpct,
                        rmident=rmident,
                        binit=binit,
                        blocks=blocks,
                    )
                )
                idx += 1

            elif sec == "GNE_DEVICE":
                data.gne_devices.append(
                    GneDevice(
                        name=_safe_str(toks[0]),
                        tokens=toks,
                    )
                )
                idx += 1

            elif sec == "INDUCTION_MACHINE":
                data.induction_machines.append(
                    InductionMachine(
                        i=_safe_int(toks[0]),
                        id=_safe_str(toks[1] if len(toks) > 1 else "1", "1"),
                        tokens=toks,
                    )
                )
                idx += 1

            elif sec == "SUBSTATION":
                data.substations.append(
                    Substation(
                        i=_safe_int(toks[0]),
                        name=_safe_str(toks[1] if len(toks) > 1 else ""),
                        tokens=toks,
                    )
                )
                idx += 1

            else:
                idx += 1

        return data


# =============================================================================
# High-Level Helper Functions
# =============================================================================


def parse_raw(file_path: str) -> PsseRawData:
    """Parse a PSS/E RAW power flow file and return PsseRawData."""
    return PsseRawParser.parse_file(file_path)


def load_raw(file_path: str) -> PsseRawData:
    """Alias for parse_raw."""
    return parse_raw(file_path)


# =============================================================================
# CLI & Demonstration
# =============================================================================

if __name__ == "__main__":
    import sys

    default_file = (
        "/usr/local/google/home/sxzhou/Downloads/2025 Series RTEP 2030 SUM_06182025.raw"
    )
    raw_path = sys.argv[1] if len(sys.argv) > 1 else default_file

    print(f"Parsing PSS/E RAW file: {raw_path}")
    t0 = time.time()
    raw_data = parse_raw(raw_path)
    t1 = time.time()

    print(f"Parsed in {t1 - t0:.2f} seconds.\n")

    print("Case Summary:")
    for k, v in raw_data.summary().items():
        print(f"  {k:25s}: {v}")

    star_buses = raw_data.get_star_buses()
    print(f"\nStar buses extracted from 3-winding transformers: {len(star_buses)}")
    if star_buses:
        first_key = next(iter(star_buses))
        print(f"  Sample star bus '{first_key}': {star_buses[first_key]}")
