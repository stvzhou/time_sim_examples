"""MATPOWER 5.0 case format classes and PSS/E RAW validation.

Follows the MATPOWER 5.0 Case Format specification:
https://matpower.org/docs/ref/matpower5.0/caseformat.html

Defines:
- Bus: MATPOWER bus matrix row representation (columns 1-13 + optional OPF cols)
- Generator: MATPOWER gen matrix row representation (columns 1-21 + optional OPF cols)
- Branch: MATPOWER branch matrix row representation (columns 1-13 + optional results cols)
- MatpowerCase: Complete MATPOWER power flow case container with conversion from PsseRawData.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

# Ensure power_flow directory is in python path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from psse_raw import PsseRawData, parse_raw


# =============================================================================
# MATPOWER Column Index Constants (MATPOWER 5.0 specification)
# =============================================================================

# Bus matrix column indices (0-indexed)
BUS_I = 0  # bus number (1 to 29997)
BUS_TYPE = 1  # bus type (1 = PQ, 2 = PV, 3 = ref/slack, 4 = isolated)
PD = 2  # real power demand (MW)
QD = 3  # reactive power demand (MVAr)
GS = 4  # shunt conductance (MW demanded at V = 1.0 p.u.)
BS = 5  # shunt susceptance (MVAr injected at V = 1.0 p.u.)
BUS_AREA = 6  # area number (1 to 100)
VM = 7  # voltage magnitude (p.u.)
VA = 8  # voltage angle (degrees)
BASE_KV = 9  # base voltage (kV)
ZONE = 10  # loss zone (1 to 999)
VMAX = 11  # maximum voltage magnitude (p.u.)
VMIN = 12  # minimum voltage magnitude (p.u.)
LAM_P = 13  # Lagrange multiplier on real power mismatch (u/MW)
LAM_Q = 14  # Lagrange multiplier on reactive power mismatch (u/MVAr)
MU_VMAX = 15  # Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
MU_VMIN = 16  # Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)

# Generator matrix column indices (0-indexed)
GEN_BUS = 0  # bus number
PG = 1  # real power output (MW)
QG = 2  # reactive power output (MVAr)
QMAX = 3  # maximum reactive power output (MVAr)
QMIN = 4  # minimum reactive power output (MVAr)
VG = 5  # voltage magnitude setpoint (p.u.)
MBASE = 6  # total MVA base of this machine, defaults to baseMVA
GEN_STATUS = 7  # machine status, > 0 = in-service, <= 0 = out-of-service
PMAX = 8  # maximum real power output (MW)
PMIN = 9  # minimum real power output (MW)
PC1 = 10  # lower real power output of PQ capability curve (MW)
PC2 = 11  # upper real power output of PQ capability curve (MW)
QC1MIN = 12  # minimum reactive power output at PC1 (MVAr)
QC1MAX = 13  # maximum reactive power output at PC1 (MVAr)
QC2MIN = 14  # minimum reactive power output at PC2 (MVAr)
QC2MAX = 15  # maximum reactive power output at PC2 (MVAr)
RAMP_AGC = 16  # ramp rate for load following/AGC (MW/min)
RAMP_10 = 17  # ramp rate for 10 minute reserves (MW)
RAMP_30 = 18  # ramp rate for 30 minute reserves (MW)
RAMP_Q = 19  # ramp rate for reactive power (2 sec reserve) (MVAr/min)
APF = 20  # area participation factor
MU_PMAX = 21  # Kuhn-Tucker multiplier on upper Pg limit (u/MW)
MU_PMIN = 22  # Kuhn-Tucker multiplier on lower Pg limit (u/MW)
MU_QMAX = 23  # Kuhn-Tucker multiplier on upper Qg limit (u/MVAr)
MU_QMIN = 24  # Kuhn-Tucker multiplier on lower Qg limit (u/MVAr)

# Branch matrix column indices (0-indexed)
F_BUS = 0  # "from" bus number
T_BUS = 1  # "to" bus number
BR_R = 2  # resistance (p.u.)
BR_X = 3  # reactance (p.u.)
BR_B = 4  # total line charging susceptance (p.u.)
RATE_A = 5  # MVA rating A (long term rating)
RATE_B = 6  # MVA rating B (short term rating)
RATE_C = 7  # MVA rating C (emergency rating)
TAP = 8  # transformer off-nominal turns ratio (0 for lines)
SHIFT = 9  # transformer phase shift angle (degrees)
BR_STATUS = 10  # branch status, 1 = in-service, 0 = out-of-service
ANGMIN = 11  # minimum angle difference, angle(Vf) - angle(Vt) (deg)
ANGMAX = 12  # maximum angle difference, angle(Vf) - angle(Vt) (deg)
PF = 13  # real power injected at "from" bus end (MW)
QF = 14  # reactive power injected at "from" bus end (MVAr)
PT = 15  # real power injected at "to" bus end (MW)
QT = 16  # reactive power injected at "to" bus end (MVAr)
MU_SF = 17  # Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
MU_ST = 18  # Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)
MU_ANGMIN = 19  # Kuhn-Tucker multiplier on lower angle diff limit (u/deg)
MU_ANGMAX = 20  # Kuhn-Tucker multiplier on upper angle diff limit (u/deg)


# =============================================================================
# Bus Class
# =============================================================================


@dataclass
class Bus:
    """MATPOWER Bus Data Format representation (bus matrix row).

    Attributes:
        bus_i: Bus number (positive integer, 1 to 29997)
        bus_type: Bus type (1 = PQ, 2 = PV, 3 = ref/slack, 4 = isolated)
        pd: Real power demand (MW)
        qd: Reactive power demand (MVAr)
        gs: Shunt conductance (MW demanded at V = 1.0 p.u.)
        bs: Shunt susceptance (MVAr injected at V = 1.0 p.u.)
        bus_area: Area number (positive integer, 1 to 100)
        vm: Voltage magnitude (p.u.)
        va: Voltage angle (degrees)
        base_kv: Base voltage (kV)
        zone: Loss zone (positive integer, 1 to 999)
        vmax: Maximum voltage magnitude (p.u.)
        vmin: Minimum voltage magnitude (p.u.)
        lam_p: Lagrange multiplier on real power mismatch (u/MW, optional)
        lam_q: Lagrange multiplier on reactive power mismatch (u/MVAr, optional)
        mu_vmax: Kuhn-Tucker multiplier on upper voltage limit (u/p.u., optional)
        mu_vmin: Kuhn-Tucker multiplier on lower voltage limit (u/p.u., optional)
    """

    bus_i: int
    bus_type: int = 1  # 1=PQ, 2=PV, 3=Slack, 4=Isolated
    pd: float = 0.0  # MW
    qd: float = 0.0  # MVAr
    gs: float = 0.0  # MW at 1.0 pu
    bs: float = 0.0  # MVAr at 1.0 pu
    bus_area: int = 1
    vm: float = 1.0  # pu
    va: float = 0.0  # deg
    base_kv: float = 100.0  # kV
    zone: int = 1
    vmax: float = 1.1  # pu
    vmin: float = 0.9  # pu
    lam_p: Optional[float] = None
    lam_q: Optional[float] = None
    mu_vmax: Optional[float] = None
    mu_vmin: Optional[float] = None

    @property
    def is_pq(self) -> bool:
        """Return True if bus is a PQ load bus."""
        return self.bus_type == 1

    @property
    def is_pv(self) -> bool:
        """Return True if bus is a PV generator bus."""
        return self.bus_type == 2

    @property
    def is_slack(self) -> bool:
        """Return True if bus is the reference/slack bus."""
        return self.bus_type == 3

    @property
    def is_isolated(self) -> bool:
        """Return True if bus is isolated/out-of-service."""
        return self.bus_type == 4

    @property
    def is_in_service(self) -> bool:
        """Return True if bus is connected and in-service."""
        return self.bus_type != 4

    def to_list(self, include_opf: bool = False) -> List[float]:
        """Convert to MATPOWER standard bus row list."""
        row = [
            float(self.bus_i),
            float(self.bus_type),
            float(self.pd),
            float(self.qd),
            float(self.gs),
            float(self.bs),
            float(self.bus_area),
            float(self.vm),
            float(self.va),
            float(self.base_kv),
            float(self.zone),
            float(self.vmax),
            float(self.vmin),
        ]
        if include_opf:
            row.extend(
                [
                    float(self.lam_p or 0.0),
                    float(self.lam_q or 0.0),
                    float(self.mu_vmax or 0.0),
                    float(self.mu_vmin or 0.0),
                ]
            )
        return row

    def to_numpy(self, include_opf: bool = False) -> np.ndarray:
        """Convert to 1D numpy array."""
        return np.array(self.to_list(include_opf=include_opf), dtype=np.float64)

    @classmethod
    def from_array(cls, row: Sequence[Union[int, float]]) -> Bus:
        """Construct Bus instance from MATPOWER bus matrix row."""
        return cls(
            bus_i=int(row[BUS_I]),
            bus_type=int(row[BUS_TYPE]),
            pd=float(row[PD]),
            qd=float(row[QD]),
            gs=float(row[GS]),
            bs=float(row[BS]),
            bus_area=int(row[BUS_AREA]),
            vm=float(row[VM]),
            va=float(row[VA]),
            base_kv=float(row[BASE_KV]),
            zone=int(row[ZONE]),
            vmax=float(row[VMAX]),
            vmin=float(row[VMIN]),
            lam_p=float(row[LAM_P]) if len(row) > LAM_P else None,
            lam_q=float(row[LAM_Q]) if len(row) > LAM_Q else None,
            mu_vmax=float(row[MU_VMAX]) if len(row) > MU_VMAX else None,
            mu_vmin=float(row[MU_VMIN]) if len(row) > MU_VMIN else None,
        )


# =============================================================================
# Generator Class
# =============================================================================


@dataclass
class Generator:
    """MATPOWER Generator Data Format representation (gen matrix row).

    Attributes:
        gen_bus: Bus number to which generator is connected
        pg: Real power output (MW)
        qg: Reactive power output (MVAr)
        qmax: Maximum reactive power output (MVAr)
        qmin: Minimum reactive power output (MVAr)
        vg: Voltage magnitude setpoint (p.u.)
        mbase: Total MVA base of this machine, defaults to baseMVA
        gen_status: Machine status (> 0 = in-service, <= 0 = out-of-service)
        pmax: Maximum real power output (MW)
        pmin: Minimum real power output (MW)
        pc1: Lower real power output of PQ capability curve (MW)
        pc2: Upper real power output of PQ capability curve (MW)
        qc1min: Minimum reactive power output at PC1 (MVAr)
        qc1max: Maximum reactive power output at PC1 (MVAr)
        qc2min: Minimum reactive power output at PC2 (MVAr)
        qc2max: Maximum reactive power output at PC2 (MVAr)
        ramp_agc: Ramp rate for load following/AGC (MW/min)
        ramp_10: Ramp rate for 10 minute reserves (MW)
        ramp_30: Ramp rate for 30 minute reserves (MW)
        ramp_q: Ramp rate for reactive power (2 sec reserve) (MVAr/min)
        apf: Area participation factor
        mu_pmax: Kuhn-Tucker multiplier on upper Pg limit (u/MW, optional)
        mu_pmin: Kuhn-Tucker multiplier on lower Pg limit (u/MW, optional)
        mu_qmax: Kuhn-Tucker multiplier on upper Qg limit (u/MVAr, optional)
        mu_qmin: Kuhn-Tucker multiplier on lower Qg limit (u/MVAr, optional)
    """

    gen_bus: int
    pg: float = 0.0  # MW
    qg: float = 0.0  # MVAr
    qmax: float = 9999.0  # MVAr
    qmin: float = -9999.0  # MVAr
    vg: float = 1.0  # pu
    mbase: float = 100.0  # MVA
    gen_status: int = 1  # 1=in-service, 0=out-of-service
    pmax: float = 9999.0  # MW
    pmin: float = -9999.0  # MW
    pc1: float = 0.0
    pc2: float = 0.0
    qc1min: float = 0.0
    qc1max: float = 0.0
    qc2min: float = 0.0
    qc2max: float = 0.0
    ramp_agc: float = 0.0
    ramp_10: float = 0.0
    ramp_30: float = 0.0
    ramp_q: float = 0.0
    apf: float = 0.0
    mu_pmax: Optional[float] = None
    mu_pmin: Optional[float] = None
    mu_qmax: Optional[float] = None
    mu_qmin: Optional[float] = None

    @property
    def is_in_service(self) -> bool:
        """Return True if generator is connected and in-service."""
        return self.gen_status > 0

    def to_list(self, include_opf: bool = False) -> List[float]:
        """Convert to MATPOWER standard gen row list."""
        row = [
            float(self.gen_bus),
            float(self.pg),
            float(self.qg),
            float(self.qmax),
            float(self.qmin),
            float(self.vg),
            float(self.mbase),
            float(self.gen_status),
            float(self.pmax),
            float(self.pmin),
            float(self.pc1),
            float(self.pc2),
            float(self.qc1min),
            float(self.qc1max),
            float(self.qc2min),
            float(self.qc2max),
            float(self.ramp_agc),
            float(self.ramp_10),
            float(self.ramp_30),
            float(self.ramp_q),
            float(self.apf),
        ]
        if include_opf:
            row.extend(
                [
                    float(self.mu_pmax or 0.0),
                    float(self.mu_pmin or 0.0),
                    float(self.mu_qmax or 0.0),
                    float(self.mu_qmin or 0.0),
                ]
            )
        return row

    def to_numpy(self, include_opf: bool = False) -> np.ndarray:
        """Convert to 1D numpy array."""
        return np.array(self.to_list(include_opf=include_opf), dtype=np.float64)

    @classmethod
    def from_array(cls, row: Sequence[Union[int, float]]) -> Generator:
        """Construct Generator instance from MATPOWER gen matrix row."""
        return cls(
            gen_bus=int(row[GEN_BUS]),
            pg=float(row[PG]),
            qg=float(row[QG]),
            qmax=float(row[QMAX]),
            qmin=float(row[QMIN]),
            vg=float(row[VG]),
            mbase=float(row[MBASE]) if len(row) > MBASE else 100.0,
            gen_status=int(row[GEN_STATUS]) if len(row) > GEN_STATUS else 1,
            pmax=float(row[PMAX]) if len(row) > PMAX else 9999.0,
            pmin=float(row[PMIN]) if len(row) > PMIN else -9999.0,
            pc1=float(row[PC1]) if len(row) > PC1 else 0.0,
            pc2=float(row[PC2]) if len(row) > PC2 else 0.0,
            qc1min=float(row[QC1MIN]) if len(row) > QC1MIN else 0.0,
            qc1max=float(row[QC1MAX]) if len(row) > QC1MAX else 0.0,
            qc2min=float(row[QC2MIN]) if len(row) > QC2MIN else 0.0,
            qc2max=float(row[QC2MAX]) if len(row) > QC2MAX else 0.0,
            ramp_agc=float(row[RAMP_AGC]) if len(row) > RAMP_AGC else 0.0,
            ramp_10=float(row[RAMP_10]) if len(row) > RAMP_10 else 0.0,
            ramp_30=float(row[RAMP_30]) if len(row) > RAMP_30 else 0.0,
            ramp_q=float(row[RAMP_Q]) if len(row) > RAMP_Q else 0.0,
            apf=float(row[APF]) if len(row) > APF else 0.0,
            mu_pmax=float(row[MU_PMAX]) if len(row) > MU_PMAX else None,
            mu_pmin=float(row[MU_PMIN]) if len(row) > MU_PMIN else None,
            mu_qmax=float(row[MU_QMAX]) if len(row) > MU_QMAX else None,
            mu_qmin=float(row[MU_QMIN]) if len(row) > MU_QMIN else None,
        )


# =============================================================================
# Branch Class
# =============================================================================


@dataclass
class Branch:
    """MATPOWER Branch Data Format representation (branch matrix row).

    Attributes:
        f_bus: "From" bus number
        t_bus: "To" bus number
        br_r: Resistance (p.u.)
        br_x: Reactance (p.u.)
        br_b: Total line charging susceptance (p.u.)
        rate_a: MVA rating A (long term rating)
        rate_b: MVA rating B (short term rating)
        rate_c: MVA rating C (emergency rating)
        tap: Transformer off-nominal turns ratio (Vf / Vt = tap / 1.0, 0 for transmission lines)
        shift: Transformer phase shift angle (degrees, positive = delay)
        br_status: Initial branch status (1 = in-service, 0 = out-of-service)
        angmin: Minimum angle difference, angle(Vf) - angle(Vt) (degrees)
        angmax: Maximum angle difference, angle(Vf) - angle(Vt) (degrees)
        pf: Real power injected at "from" bus end (MW, optional)
        qf: Reactive power injected at "from" bus end (MVAr, optional)
        pt: Real power injected at "to" bus end (MW, optional)
        qt: Reactive power injected at "to" bus end (MVAr, optional)
        mu_sf: Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA, optional)
        mu_st: Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA, optional)
        mu_angmin: Kuhn-Tucker multiplier on lower angle diff limit (u/deg, optional)
        mu_angmax: Kuhn-Tucker multiplier on upper angle diff limit (u/deg, optional)
    """

    f_bus: int
    t_bus: int
    br_r: float = 0.0  # pu
    br_x: float = 0.0001  # pu
    br_b: float = 0.0  # pu
    rate_a: float = 0.0  # MVA
    rate_b: float = 0.0  # MVA
    rate_c: float = 0.0  # MVA
    tap: float = 0.0  # 0.0 for lines, ratio for transformers
    shift: float = 0.0  # degrees
    br_status: int = 1  # 1=in-service, 0=out-of-service
    angmin: float = -360.0  # degrees
    angmax: float = 360.0  # degrees
    pf: Optional[float] = None
    qf: Optional[float] = None
    pt: Optional[float] = None
    qt: Optional[float] = None
    mu_sf: Optional[float] = None
    mu_st: Optional[float] = None
    mu_angmin: Optional[float] = None
    mu_angmax: Optional[float] = None

    @property
    def is_transformer(self) -> bool:
        """Return True if branch is a transformer (tap != 0 or shift != 0)."""
        return self.tap != 0.0 or self.shift != 0.0

    @property
    def is_line(self) -> bool:
        """Return True if branch is a transmission line."""
        return not self.is_transformer

    @property
    def is_in_service(self) -> bool:
        """Return True if branch is in-service."""
        return self.br_status == 1

    def to_list(self, include_results: bool = False) -> List[float]:
        """Convert to MATPOWER standard branch row list."""
        row = [
            float(self.f_bus),
            float(self.t_bus),
            float(self.br_r),
            float(self.br_x),
            float(self.br_b),
            float(self.rate_a),
            float(self.rate_b),
            float(self.rate_c),
            float(self.tap),
            float(self.shift),
            float(self.br_status),
            float(self.angmin),
            float(self.angmax),
        ]
        if include_results:
            row.extend(
                [
                    float(self.pf or 0.0),
                    float(self.qf or 0.0),
                    float(self.pt or 0.0),
                    float(self.qt or 0.0),
                    float(self.mu_sf or 0.0),
                    float(self.mu_st or 0.0),
                    float(self.mu_angmin or 0.0),
                    float(self.mu_angmax or 0.0),
                ]
            )
        return row

    def to_numpy(self, include_results: bool = False) -> np.ndarray:
        """Convert to 1D numpy array."""
        return np.array(self.to_list(include_results=include_results), dtype=np.float64)

    @classmethod
    def from_array(cls, row: Sequence[Union[int, float]]) -> Branch:
        """Construct Branch instance from MATPOWER branch matrix row."""
        return cls(
            f_bus=int(row[F_BUS]),
            t_bus=int(row[T_BUS]),
            br_r=float(row[BR_R]),
            br_x=float(row[BR_X]),
            br_b=float(row[BR_B]),
            rate_a=float(row[RATE_A]),
            rate_b=float(row[RATE_B]),
            rate_c=float(row[RATE_C]),
            tap=float(row[TAP]),
            shift=float(row[SHIFT]),
            br_status=int(row[BR_STATUS]),
            angmin=float(row[ANGMIN]) if len(row) > ANGMIN else -360.0,
            angmax=float(row[ANGMAX]) if len(row) > ANGMAX else 360.0,
            pf=float(row[PF]) if len(row) > PF else None,
            qf=float(row[QF]) if len(row) > QF else None,
            pt=float(row[PT]) if len(row) > PT else None,
            qt=float(row[QT]) if len(row) > QT else None,
            mu_sf=float(row[MU_SF]) if len(row) > MU_SF else None,
            mu_st=float(row[MU_ST]) if len(row) > MU_ST else None,
            mu_angmin=float(row[MU_ANGMIN]) if len(row) > MU_ANGMIN else None,
            mu_angmax=float(row[MU_ANGMAX]) if len(row) > MU_ANGMAX else None,
        )


# =============================================================================
# MatpowerCase Container
# =============================================================================


@dataclass
class MatpowerCase:
    """MATPOWER 5.0 Case container holding baseMVA, bus, gen, and branch matrices.

    Attributes:
        version: Case format version (default: '2')
        baseMVA: System base MVA (default: 100.0)
        bus: List of Bus instances
        gen: List of Generator instances
        branch: List of Branch instances
        gencost: Optional generator cost matrix
    """

    version: str = "2"
    baseMVA: float = 100.0
    bus: List[Bus] = field(default_factory=list)
    gen: List[Generator] = field(default_factory=list)
    branch: List[Branch] = field(default_factory=list)
    gencost: Optional[np.ndarray] = None
    star_buses: Dict[int, List[int]] = field(default_factory=lambda: {})

    def to_dict(self) -> Dict[str, Any]:
        """Export case as standard MATPOWER dictionary with numpy matrices."""
        bus_mat = (
            np.array([b.to_list() for b in self.bus], dtype=np.float64)
            if self.bus
            else np.empty((0, 13))
        )
        gen_mat = (
            np.array([g.to_list() for g in self.gen], dtype=np.float64)
            if self.gen
            else np.empty((0, 21))
        )
        branch_mat = (
            np.array([br.to_list() for br in self.branch], dtype=np.float64)
            if self.branch
            else np.empty((0, 13))
        )

        d: Dict[str, Any] = {
            "version": self.version,
            "baseMVA": self.baseMVA,
            "bus": bus_mat,
            "gen": gen_mat,
            "branch": branch_mat,
        }
        if self.gencost is not None:
            d["gencost"] = self.gencost
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MatpowerCase:
        """Construct MatpowerCase from MATPOWER dictionary (e.g. from pypower / loadmat)."""
        base_mva = float(d.get("baseMVA", 100.0))
        version = str(d.get("version", "2"))

        buses = [Bus.from_array(row) for row in d.get("bus", [])]
        generators = [Generator.from_array(row) for row in d.get("gen", [])]
        branches = [Branch.from_array(row) for row in d.get("branch", [])]
        gencost = d.get("gencost")

        return cls(
            version=version,
            baseMVA=base_mva,
            bus=buses,
            gen=generators,
            branch=branches,
            gencost=gencost,
        )

    @classmethod
    def from_psse(cls, raw: PsseRawData) -> MatpowerCase:
        """Convert a parsed PsseRawData instance into a MATPOWER 5.0 Case."""
        base_mva = raw.case_id.sbase if raw.case_id.sbase > 0 else 100.0

        # 1. Aggregate loads by bus (MW and MVAr)
        bus_pd: Dict[int, float] = {}
        bus_qd: Dict[int, float] = {}
        for load in raw.loads:
            if load.is_in_service:
                bus_pd[load.i] = bus_pd.get(load.i, 0.0) + load.pl
                bus_qd[load.i] = bus_qd.get(load.i, 0.0) + load.ql

        # 2. Aggregate fixed shunts by bus (MW and MVAr at 1.0 pu)
        bus_gs: Dict[int, float] = {}
        bus_bs: Dict[int, float] = {}
        for shunt in raw.fixed_shunts:
            if shunt.is_in_service:
                bus_gs[shunt.i] = bus_gs.get(shunt.i, 0.0) + shunt.gl
                bus_bs[shunt.i] = bus_bs.get(shunt.i, 0.0) + shunt.bl

        # 3. Add initial switched shunt admittance (binit)
        for sw in raw.switched_shunts:
            if sw.is_in_service:
                bus_bs[sw.i] = bus_bs.get(sw.i, 0.0) + sw.binit

        # 4. Construct Bus list
        mat_buses: List[Bus] = []
        for b in raw.buses.values():
            mat_buses.append(
                Bus(
                    bus_i=b.i,
                    bus_type=b.ide,
                    pd=bus_pd.get(b.i, 0.0),
                    qd=bus_qd.get(b.i, 0.0),
                    gs=bus_gs.get(b.i, 0.0),
                    bs=bus_bs.get(b.i, 0.0),
                    bus_area=b.area,
                    vm=b.vm,
                    va=b.va,
                    base_kv=b.baskv,
                    zone=b.zone,
                    vmax=b.nvhi if b.nvhi > 0 else 1.1,
                    vmin=b.nvlo if b.nvlo > 0 else 0.9,
                )
            )

        # 5. Construct Generator list
        mat_gens: List[Generator] = []
        for g in raw.generators:
            mat_gens.append(
                Generator(
                    gen_bus=g.i,
                    pg=g.pg,
                    qg=g.qg,
                    qmax=g.qt,
                    qmin=g.qb,
                    vg=g.vs,
                    mbase=g.mbase if g.mbase > 0 else base_mva,
                    gen_status=g.stat,
                    pmax=g.pt,
                    pmin=g.pb,
                )
            )

        # 6. Construct Branch list (AC lines + 2W transformers + 3W transformers)
        mat_branches: List[Branch] = []

        # AC transmission lines
        for br in raw.branches:
            mat_branches.append(
                Branch(
                    f_bus=br.i,
                    t_bus=br.j,
                    br_r=br.r,
                    br_x=br.x,
                    br_b=br.b,
                    rate_a=br.ratea,
                    rate_b=br.rateb,
                    rate_c=br.ratec,
                    tap=0.0,
                    shift=0.0,
                    br_status=br.st,
                    angmin=-360.0,
                    angmax=360.0,
                )
            )

        # 2-Winding Transformers
        for t2 in raw.transformers_2w:
            # Turns ratio: ratio = windv1 / windv2 (if windv2 == 0, default 1.0)
            w2_ratio = t2.wdg2.windv if t2.wdg2.windv > 0 else 1.0
            tap = t2.wdg1.windv / w2_ratio if t2.wdg1.windv > 0 else 1.0
            shift = t2.wdg1.ang - t2.wdg2.ang

            if t2.cz == 1:
                r1_2 = t2.r1_2
                x1_2 = t2.x1_2
            elif t2.cz == 2:
                r1_2 = t2.r1_2 / t2.sbase1_2 * base_mva
                x1_2 = t2.x1_2 / t2.sbase1_2 * base_mva
            elif t2.cz == 3:
                rw = t2.r1_2 / t2.sbase1_2 / 1e6
                xw = np.sqrt(t2.x1_2**2 - rw**2)
                r1_2 = rw * base_mva / t2.sbase1_2
                x1_2 = xw * base_mva / t2.sbase1_2
            else:
                raise ValueError(f"Unsupported transformer impedance format: {t2.cz}")

            mat_branches.append(
                Branch(
                    f_bus=t2.i,
                    t_bus=t2.j,
                    br_r=r1_2,
                    br_x=x1_2,
                    br_b=t2.mag2,  # Magnetizing susceptance
                    rate_a=t2.wdg1.rata,
                    rate_b=t2.wdg1.ratb,
                    rate_c=t2.wdg1.ratc,
                    tap=tap,
                    shift=shift,
                    br_status=t2.stat,
                    angmin=-360.0,
                    angmax=360.0,
                )
            )

        # 3-Winding Transformers (Star Equivalent Model)
        # Allocate star bus IDs starting above highest existing bus ID
        max_bus_id = max((b.i for b in raw.buses.values()), default=0)
        star_bus_counter = max_bus_id + 1
        star_buses = defaultdict(list)

        for t3 in raw.transformers_3w:
            star_bus_id = star_bus_counter
            star_buses[star_bus_id].extend([t3.i, t3.j, t3.k])
            star_bus_counter += 1

            # Determine base kV for star bus from winding 1
            bus1_ref = raw.buses.get(t3.i)
            base_kv = bus1_ref.baskv if bus1_ref else 100.0

            # Add internal star bus to bus list
            mat_buses.append(
                Bus(
                    bus_i=star_bus_id,
                    bus_type=1 if t3.stat != 0 else 4,  # PQ bus
                    pd=0.0,
                    qd=0.0,
                    gs=0.0,
                    bs=0.0,
                    bus_area=bus1_ref.area if bus1_ref else 1,
                    vm=t3.vmstar if t3.vmstar > 0 else 1.0,
                    va=t3.anstar,
                    base_kv=base_kv,
                    zone=bus1_ref.zone if bus1_ref else 1,
                    vmax=1.1,
                    vmin=0.9,
                )
            )
            sbase = raw.case_id.sbase
            if t3.cz == 1:
                r1_2 = t3.r1_2
                x1_2 = t3.x1_2
                r2_3 = t3.r2_3
                x2_3 = t3.x2_3
                r3_1 = t3.r3_1
                x3_1 = t3.x3_1
            elif t3.cz == 2:
                r1_2 = t3.r1_2 / t3.sbase1_2 * base_mva
                x1_2 = t3.x1_2 / t3.sbase1_2 * base_mva
                r2_3 = t3.r2_3 / t3.sbase2_3 * base_mva
                x2_3 = t3.x2_3 / t3.sbase2_3 * base_mva
                r3_1 = t3.r3_1 / t3.sbase3_1 * base_mva
                x3_1 = t3.x3_1 / t3.sbase3_1 * base_mva
            elif t3.cz == 3:
                r1_2 = t3.r1_2 / t3.sbase1_2 / 1e6
                r2_3 = t3.r2_3 / t3.sbase2_3 / 1e6
                r3_1 = t3.r3_1 / t3.sbase3_1 / 1e6
                x1_2 = np.sqrt(t3.x1_2**2 - r1_2**2)
                x2_3 = np.sqrt(t3.x2_3**2 - r2_3**2)
                x3_1 = np.sqrt(t3.x3_1**2 - r3_1**2)
                r1_2 = r1_2 * base_mva / t3.sbase1_2
                r2_3 = r2_3 * base_mva / t3.sbase2_3
                r3_1 = r3_1 * base_mva / t3.sbase3_1
                x1_2 = x1_2 * base_mva / t3.sbase1_2
                x2_3 = x2_3 * base_mva / t3.sbase2_3
                x3_1 = x3_1 * base_mva / t3.sbase3_1
            else:
                raise ValueError(f"Unsupported transformer impedance format: {t3.cz}")

            r1 = 0.5 * (r1_2 + r3_1 - r2_3)
            x1 = 0.5 * (x1_2 + x3_1 - x2_3)
            r2 = 0.5 * (r1_2 + r2_3 - r3_1)
            x2 = 0.5 * (x1_2 + x2_3 - x3_1)
            r3 = 0.5 * (r2_3 + r3_1 - r1_2)
            x3 = 0.5 * (x2_3 + x3_1 - x1_2)

            # Leg 1: Bus i <-> Star Bus
            mat_branches.append(
                Branch(
                    f_bus=t3.i,
                    t_bus=star_bus_id,
                    br_r=r1,
                    br_x=x1,
                    br_b=0.0,
                    rate_a=t3.wdg1.rata,
                    rate_b=t3.wdg1.ratb,
                    rate_c=t3.wdg1.ratc,
                    tap=t3.wdg1.windv if t3.wdg1.windv > 0 else 1.0,
                    shift=t3.wdg1.ang,
                    br_status=t3.stat in [1, 3, 4],
                )
            )

            # Leg 2: Bus j <-> Star Bus
            mat_branches.append(
                Branch(
                    f_bus=t3.j,
                    t_bus=star_bus_id,
                    br_r=r2,
                    br_x=x2,
                    br_b=0.0,
                    rate_a=t3.wdg2.rata,
                    rate_b=t3.wdg2.ratb,
                    rate_c=t3.wdg2.ratc,
                    tap=t3.wdg2.windv if t3.wdg2.windv > 0 else 1.0,
                    shift=t3.wdg2.ang,
                    br_status=t3.stat in [1, 2, 4],
                )
            )

            # Leg 3: Bus k <-> Star Bus
            mat_branches.append(
                Branch(
                    f_bus=t3.k,
                    t_bus=star_bus_id,
                    br_r=r3,
                    br_x=x3,
                    br_b=0.0,
                    rate_a=t3.wdg3.rata,
                    rate_b=t3.wdg3.ratb,
                    rate_c=t3.wdg3.ratc,
                    tap=t3.wdg3.windv if t3.wdg3.windv > 0 else 1.0,
                    shift=t3.wdg3.ang,
                    br_status=t3.stat in [1, 2, 3],
                )
            )

        return cls(
            version="2",
            baseMVA=base_mva,
            bus=mat_buses,
            gen=mat_gens,
            branch=mat_branches,
            star_buses=star_buses,
        )

    def summary(self) -> Dict[str, Any]:
        """Summarize MATPOWER case statistics."""
        total_p_gen = sum(g.pg for g in self.gen if g.is_in_service)
        total_q_gen = sum(g.qg for g in self.gen if g.is_in_service)
        total_p_load = sum(b.pd for b in self.bus if b.is_in_service)
        total_q_load = sum(b.qd for b in self.bus if b.is_in_service)
        total_shunts = sum(b.bs for b in self.bus if b.is_in_service)

        num_slack = sum(1 for b in self.bus if b.is_slack)
        num_pv = sum(1 for b in self.bus if b.is_pv)
        num_pq = sum(1 for b in self.bus if b.is_pq)
        num_trans = sum(1 for br in self.branch if br.is_transformer)
        num_lines = sum(1 for br in self.branch if br.is_line)

        return {
            "baseMVA": self.baseMVA,
            "buses_total": len(self.bus),
            "buses_slack": num_slack,
            "buses_pv": num_pv,
            "buses_pq": num_pq,
            "generators_total": len(self.gen),
            "generators_in_service": sum(1 for g in self.gen if g.is_in_service),
            "branches_total": len(self.branch),
            "transmission_lines": num_lines,
            "transformers": num_trans,
            "total_gen_mw": round(total_p_gen, 2),
            "total_load_mw": round(total_p_load, 2),
            "shunts_mvar": round(total_shunts, 2),
            "total_gen_mvar": round(total_q_gen, 2),
            "total_load_mvar": round(total_q_load, 2),
        }


# =============================================================================
# Validation and Execution
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    raw_file = (
        "/usr/local/google/home/sxzhou/Downloads/2025 Series RTEP 2030 SUM_06182025.raw"
    )

    print(f"1. Parsing RAW file: {raw_file}")
    raw_data = parse_raw(raw_file)

    print("2. Converting to MATPOWER 5.0 Case...")
    mpc = MatpowerCase.from_psse(raw_data)

    print("\nMATPOWER Case Summary:")
    for k, v in mpc.summary().items():
        print(f"  {k:25s}: {v}")

    print("\nFirst 3 Buses (MATPOWER format):")
    for b in mpc.bus[:3]:
        print(
            f"  Bus {b.bus_i}: type={b.bus_type}, Pd={b.pd} MW, Qd={b.qd} MVAr, Vm={b.vm:.3f} pu, Va={b.va:.2f} deg, baseKV={b.base_kv} kV"
        )

    print("\nFirst 3 Generators (MATPOWER format):")
    for g in mpc.gen[:3]:
        print(
            f"  Gen at Bus {g.gen_bus}: Pg={g.pg} MW, Qg={g.qg} MVAr, Qmax={g.qmax}, Qmin={g.qmin}, Vg={g.vg} pu, status={g.gen_status}"
        )

    print("\nFirst 3 Branches (MATPOWER format):")
    for br in mpc.branch[:3]:
        br_type = "Transformer" if br.is_transformer else "Line"
        print(
            f"  {br_type} {br.f_bus} -> {br.t_bus}: r={br.br_r:.4f}, x={br.br_x:.4f}, b={br.br_b:.4f}, tap={br.tap}, shift={br.shift} deg, status={br.br_status}"
        )

    # Verify dictionary export with numpy matrices
    d = mpc.to_dict()
    print("\nMATPOWER dictionary matrix shapes:")
    print(f"  bus matrix:    {d['bus'].shape}")
    print(f"  gen matrix:    {d['gen'].shape}")
    print(f"  branch matrix: {d['branch'].shape}")

    # from pypower.api import runpf
    # from pypower.ppoption import ppoption

    # ppc = mpc.to_dict()

    # ppopt = ppoption(
    #     MODEL="AC",  # AC power flow model
    #     PF_ALG=1,  # 1 = Newton-Raphson ('NR')
    #     PF_TOL=1e-8,  # Convergence tolerance
    #     PF_MAX_IT=10,  # Maximum iteration limit
    #     ENFORCE_Q_LIMS=0,  # 0 = Do not enforce Q limits initially
    #     VERBOSE=2,  # 2 = Print detailed progress
    #     OUT_ALL=1,  # 1 = Print all output sections
    # )
    # results, success = runpf(ppc, ppopt)
    # print(f"\nPower flow converged: {bool(success)}")
