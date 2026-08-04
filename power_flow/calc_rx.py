import numpy as np
from enum import Enum
import math
from dataclasses import dataclass


class Model(Enum):
    MATPOWER = "MATPOWER"
    TARA = "TARA"


@dataclass
class Flow:
    """Active (P) and Reactive (Q) power flow representation.

    Attributes:
        p: Active power (MW)
        q: Reactive power (MVAr)
    """

    p: float = 0.0
    q: float = 0.0

    # @property
    # def s(self) -> float:
    #     """Apparent power magnitude (MVA) S = sqrt(P^2 + Q^2)."""
    #     return math.sqrt(self.p**2 + self.q**2)

    # def __add__(self, other: Flow) -> Flow:
    #     return Flow(self.p + other.p, self.q + other.q)

    # def __sub__(self, other: Flow) -> Flow:
    #     return Flow(self.p - other.p, self.q - other.q)

    # def __neg__(self) -> Flow:
    #     return Flow(-self.p, -self.q)

    # def is_zero(self, tol: float = 1e-8) -> bool:
    #     """Return True if both P and Q are within tolerance of zero."""
    #     return abs(self.p) <= tol and abs(self.q) <= tol

    # def to_tuple(self) -> Tuple[float, float]:
    #     """Return (p, q) tuple."""
    #     return (self.p, self.q)

    # def to_dict(self) -> Dict[str, float]:
    #     """Return dictionary representation."""
    #     return {"p": self.p, "q": self.q, "s": self.s}


def calc_flow_into_branch(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b_total: float = 0.0,
    shift: float = 0.0,
    ratio: float = 1.0,
    gfrom: float = 0,
    bfrom: float = 0,
    gto: float = 0,
    bto: float = 0,
    base_mva: float = 100.0,
) -> Flow:
    """Calculate active and reactive power flow INTO a branch at the 'from' bus.
    https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/loadflow/loadflow.html

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
    if r**2 + x**2 == 0:
        return Flow(0, 0)

    a1 = np.deg2rad(shift)
    b = b_total / 2.0
    gkm = r / (r**2 + x**2)
    bkm = -x / (r**2 + x**2)
    theta = np.deg2rad(vsa) - np.deg2rad(vra) - a1
    if ratio == 0:
        ratio = 1
    vsm = vsm / ratio

    # Active power flow into branch at from bus
    ps = (
        vsm**2 * gkm - vsm * vrm * gkm * np.cos(theta) - vsm * vrm * bkm * np.sin(theta)
    )

    # Reactive power flow into branch at from bus
    qs = (
        -(vsm**2) * (bkm + b)
        + vsm * vrm * bkm * np.cos(theta)
        - vsm * vrm * gkm * np.sin(theta)
    )

    vsm = vsm * ratio
    ps += gfrom * vsm * vsm
    qs -= bfrom * vsm * vsm

    return Flow(float(ps * base_mva), float(qs * base_mva))


def calc_flow_from_branch(
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    r: float,
    x: float,
    b_total: float = 0.0,
    shift: float = 0.0,
    ratio: float = 1.0,
    gfrom: float = 0,
    bfrom: float = 0,
    gto: float = 0,
    bto: float = 0,
    base_mva: float = 100.0,
    model: Model = Model.MATPOWER,
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
    if r**2 + x**2 == 0:
        return Flow(0, 0)

    a1 = np.deg2rad(shift)
    b = b_total / 2.0
    gkm = r / (r**2 + x**2)
    bkm = -x / (r**2 + x**2)

    # Angle difference from sending to receiving end
    theta = np.deg2rad(vsa) - np.deg2rad(vra) - a1

    if ratio == 0:
        ratio = 1

    # ratio on from or to bus will lead to significant power flow difference, MATPOWER only support ratio on from bus
    original_vrm = vrm
    if model == Model.TARA:
        vrm = vrm * ratio
    elif model == Model.MATPOWER:
        vsm = vsm / ratio

    # Active power flow injected at receiving bus from line
    pr = (vrm**2) * gkm - vsm * vrm * (gkm * np.cos(theta) - bkm * np.sin(theta))

    # Reactive power flow injected at receiving bus from line
    qr = -(vrm**2) * (bkm + b) + vsm * vrm * (bkm * np.cos(theta) + gkm * np.sin(theta))

    pr -= gto * (original_vrm**2)
    qr -= bto * (original_vrm**2)

    return Flow(float(pr * base_mva), float(qr * base_mva))


def calculate_rx_by_pq_from_branch(
    pr: float,
    qr: float,
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    ratio: float,
    b: float,
    gfrom: float = 0,
    bfrom: float = 0,
    gto: float = 0,
    bto: float = 0,
    shift: float = 0,
    sbase: float = 100,
    model: Model = Model.MATPOWER,
):
    """
    Calculates r and x fromreceiving end power flows.

    Parameters:
    pr, qr : Active and reactive power flows entering branch at 'from' bus
    vsm, vrm : Voltage magnitudes at sending and receiving buses
    theta : Phase angle difference (va_from - va_to) in radians
    ratio : Transformer off-nominal tap ratio
    b : Line shunt susceptance (half line charging B)
    """
    # Intermediate term shortcuts
    pr = pr / sbase
    qr = qr / sbase
    pr += gto * vrm * vrm
    qr += bto * vrm * vrm

    b = b / 2
    theta = np.radians(vsa - vra - shift)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if model == Model.TARA:
        vrm = vrm * ratio
    elif model == Model.MATPOWER:
        vsm = vsm / ratio

    A_coeff = -vsm * vrm * cos_t + vrm**2
    B_coeff = vsm * vrm * sin_t
    K1 = pr

    C_coeff = vrm * vsm * sin_t
    D_coeff = (vrm * vsm * cos_t) - vrm**2
    K2 = qr + (vrm**2) * b

    M = np.array([[A_coeff, B_coeff], [C_coeff, D_coeff]])

    K = np.array([K1, K2])
    gkm, bkm = np.linalg.solve(M, K)

    denom = gkm**2 + bkm**2
    r = gkm / denom
    x = -bkm / denom

    return r, x


def calculate_rx_by_pq_into_branch(
    ps: float,
    qs: float,
    vsm: float,
    vrm: float,
    vsa: float,
    vra: float,
    ratio: float,
    b: float,
    gfrom: float = 0,
    bfrom: float = 0,
    gto: float = 0,
    bto: float = 0,
    shift=0,
    sbase=100,
):
    """
    Calculates r and x from sending end power flows.

    Parameters:
    pr, qr : Active and reactive power flows entering branch at 'from' bus
    vsm, vrm : Voltage magnitudes at sending and receiving buses
    theta : Phase angle difference (va_from - va_to) in radians
    ratio : Transformer off-nominal tap ratio
    b : Line shunt susceptance (half line charging B)
    """
    # Intermediate term shortcuts
    ps = ps / sbase
    qs = qs / sbase
    ps -= gfrom * vsm * vsm
    qs += bfrom * vsm * vsm

    b = b / 2
    theta = np.radians(vsa - vra - shift)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if ratio == 0:
        ratio = 1

    vsm = vsm / ratio

    A_coeff = -vsm * vrm * cos_t + vsm**2
    B_coeff = -vsm * vrm * sin_t
    K1 = ps

    C_coeff = -vrm * vsm * sin_t
    D_coeff = vrm * vsm * cos_t - vsm**2
    K2 = qs + (vsm**2) * b

    M = np.array([[A_coeff, B_coeff], [C_coeff, D_coeff]])

    K = np.array([K1, K2])
    gkm, bkm = np.linalg.solve(M, K)

    denom = gkm**2 + bkm**2
    r = gkm / denom
    x = -bkm / denom

    return r, x


def test_case_into_line():
    vsm = 1.04068
    vrm = 1.04545
    vsa = -27.114
    vra = -28.432
    ratio = 1.0
    b = 1.003
    pr = 230.22
    qr = -109.64

    r_calc, x_calc = calculate_rx_by_pq_into_branch(
        pr, qr, vsm, vrm, vsa, vra, ratio, b
    )
    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.00055
    x = 0.01074
    assert math.isclose(r, r_calc, rel_tol=1e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)
    flow = calc_flow_into_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
    )
    assert math.isclose(flow.p, pr, rel_tol=1e-2)
    assert math.isclose(flow.q, qr, rel_tol=1e-2)


def test_case_from_line():
    vsm = 1.04068
    vrm = 1.04501
    vsa = -27.114
    vra = -25.56
    ratio = 1.0
    b = 1.0865
    pr = 246
    qr = -30.8

    r_calc, x_calc = calculate_rx_by_pq_from_branch(
        pr, qr, vsm, vrm, vsa, vra, ratio, b
    )

    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.000607
    x = 0.012024
    assert math.isclose(r, r_calc, rel_tol=1e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)

    flow = calc_flow_from_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
        shift=0,
    )
    assert math.isclose(flow.p, pr, rel_tol=1e-2)
    assert math.isclose(flow.q, qr, rel_tol=1e-2)


def test_case_from_line_w_shift():
    vsm = 1.01897
    vrm = 1.0
    vsa = -18.648
    vra = 18.339
    ratio = 1.03774
    shift = -30
    b = 0
    pr = 538.36
    qr = 109.81
    gfrom = 0.0029
    bfrom = -0.0016
    gto = 0
    bto = 0

    r_calc, x_calc = calculate_rx_by_pq_from_branch(
        pr=pr,
        qr=qr,
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        b=b,
        shift=shift,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
        model=Model.TARA,
    )

    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.0002
    x = 0.02393
    assert math.isclose(r, r_calc, rel_tol=3e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)

    flow = calc_flow_from_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
        shift=shift,
        model=Model.TARA,
    )
    assert math.isclose(flow.p, pr, rel_tol=1e-2)
    assert math.isclose(flow.q, qr, rel_tol=1e-2)


def test_case_into_line_w_shift():
    vsm = 1.025
    vrm = 0.99695
    vsa = -20.01
    vra = 17.357
    ratio = 1.025
    shift = -30
    b = 0
    ps = -193
    qs = 22.04
    gfrom = 0
    bfrom = 0
    gto = 0
    bto = 0

    r_calc, x_calc = calculate_rx_by_pq_into_branch(
        ps=ps,
        qs=qs,
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        b=b,
        shift=shift,
    )

    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.0017
    x = 0.066
    assert math.isclose(r, r_calc, rel_tol=1e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)

    flow = calc_flow_into_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
        shift=shift,
    )
    assert math.isclose(flow.p, ps, rel_tol=1e-2)
    assert math.isclose(flow.q, qs, rel_tol=1e-2)


def test_case_into_line_w_shunt():
    vsm = 1.04979
    vrm = 1.01
    vsa = -65.263
    vra = -68.698
    ratio = 1.0
    shift = 0
    b = 0
    ps = 331.86
    qs = 226.62
    gfrom = 0.0133
    bfrom = -0.0135
    gto = 0
    bto = 0

    r_calc, x_calc = calculate_rx_by_pq_into_branch(
        ps=ps,
        qs=qs,
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        b=b,
        shift=shift,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
    )

    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.00008
    x = 0.01928
    assert math.isclose(r, r_calc, rel_tol=1e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)

    flow = calc_flow_into_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
        shift=shift,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
    )
    assert math.isclose(flow.p, ps, rel_tol=1e-2)
    assert math.isclose(flow.q, qs, rel_tol=1e-2)


def test_case_from_line_w_shunt():
    vsm = 1.03492
    vrm = 1.03263
    vsa = -9.648
    vra = -8.793
    ratio = 1.0
    shift = 0
    b = 0.35372
    pr = 76.61
    qr = -142.94
    gfrom = 0
    bfrom = 0
    gto = 0
    bto = 1.0

    r_calc, x_calc = calculate_rx_by_pq_from_branch(
        pr=pr,
        qr=qr,
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        b=b,
        shift=shift,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
    )

    print(f"Calculated R and X: {r_calc},{x_calc}")
    r = 0.00171
    x = 0.02042
    assert math.isclose(r, r_calc, rel_tol=1e-2)
    assert math.isclose(x, x_calc, rel_tol=1e-2)

    flow = calc_flow_from_branch(
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        r=r,
        x=x,
        b_total=b,
        shift=shift,
        gfrom=gfrom,
        bfrom=bfrom,
        gto=gto,
        bto=bto,
    )
    assert math.isclose(flow.p, pr, rel_tol=1e-2)
    assert math.isclose(flow.q, qr, rel_tol=1e-2)


if __name__ == "__main__":
    test_case_into_line()
    test_case_from_line()
    test_case_from_line_w_shift()
    test_case_into_line_w_shift()
    test_case_from_line_w_shunt()
    test_case_into_line_w_shunt()
