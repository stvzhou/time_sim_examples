import numpy as np
from validation import calc_flow_from_branch, calc_flow_into_branch


def calculate_rx_by_pq_from_line(
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
    theta = np.radians(vsa - vra)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vrm = vrm * ratio

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


def calculate_rx_by_pq_into_line(
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

    v_from = vsm / ratio

    A_coeff = -v_from * vrm * cos_t + v_from**2
    B_coeff = -v_from * vrm * sin_t
    K1 = ps

    C_coeff = -vrm * v_from * sin_t
    D_coeff = vrm * v_from * cos_t - v_from**2
    K2 = qs + (v_from**2) * b

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

    r, x = calculate_rx_by_pq_into_line(pr, qr, vsm, vrm, vsa, vra, ratio, b)
    print(f"Calculated R and X: {r},{x}")
    r = 0.00055
    x = 0.01074
    print(
        calc_flow_into_branch(
            vsm=vsm,
            vrm=vrm,
            vsa=vsa,
            vra=vra,
            ratio=ratio,
            r=r,
            x=x,
            b_total=b,
        )
    )


def test_case_from_line():
    vsm = 1.04068
    vrm = 1.04501
    vsa = -27.114
    vra = -25.56
    ratio = 1.0
    b = 1.0865
    pr = 246
    qr = -30.8

    r, x = calculate_rx_by_pq_from_line(pr, qr, vsm, vrm, vsa, vra, ratio, b)

    print(f"Calculated R and X: {r},{x}")
    r = 0.000607
    x = 0.012024
    print(
        calc_flow_from_branch(
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
    )


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

    r, x = calculate_rx_by_pq_from_line(
        pr=pr,
        qr=qr,
        vsm=vsm,
        vrm=vrm,
        vsa=vsa,
        vra=vra,
        ratio=ratio,
        b=b,
        shift=shift,
    )

    print(f"Calculated R and X: {r},{x}")
    r = 0.0002
    x = 0.02393
    print(
        calc_flow_from_branch(
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
    )


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

    r, x = calculate_rx_by_pq_into_line(
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

    print(f"Calculated R and X: {r},{x}")
    r = 0.0017
    x = 0.066
    print(
        calc_flow_into_branch(
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
    )


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

    r, x = calculate_rx_by_pq_into_line(
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

    print(f"Calculated R and X: {r},{x}")
    r = 0.00008
    x = 0.01928
    print(
        calc_flow_into_branch(
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
    )


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

    r, x = calculate_rx_by_pq_from_line(
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

    print(f"Calculated R and X: {r},{x}")
    r = 0.00171
    x = 0.02042
    print(
        calc_flow_from_branch(
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
    )


test_case_into_line_w_shunt()
