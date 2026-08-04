import numpy as np
from validation import calc_flow_from_branch, calc_flow_into_branch


def calculate_gkm_bkm(pr, qr, vsm, vrm, vsa, vra, ratio, b):
    """
    Calculates gkm and bkm from sending/receiving end power flows.

    Parameters:
    pr, qr : Active and reactive power flows entering branch at 'from' bus
    vsm, vrm : Voltage magnitudes at sending and receiving buses
    theta : Phase angle difference (va_from - va_to) in radians
    ratio : Transformer off-nominal tap ratio
    b : Line shunt susceptance (half line charging B)
    """
    # Intermediate term shortcuts
    b = b / 2
    theta = np.radians(vsa - vra)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Linear system coefficients:
    # Eq 1: A * gkm + B_coeff * bkm = K1
    # Eq 2: C_coeff * gkm + D_coeff * bkm = K2
    v_from = vsm / ratio

    A_coeff = -v_from * vrm * cos_t + vrm**2
    B_coeff = v_from * vrm * sin_t
    K1 = pr

    C_coeff = vrm * v_from * sin_t
    D_coeff = (vrm * v_from * cos_t) - vrm**2
    K2 = qr + (vrm**2) * b

    # Solve 2x2 linear system matrix [M] * [gkm, bkm]^T = [K1, K2]^T
    M = np.array([[A_coeff, B_coeff], [C_coeff, D_coeff]])

    K = np.array([K1, K2])

    # Direct matrix solve
    gkm, bkm = np.linalg.solve(M, K)

    denom = gkm**2 + bkm**2
    r = gkm / denom
    x = -bkm / denom

    return r, x


# --- Test with known values ---
# Example inputs
vsm, vrm = 1.04068, 1.04501
vsa, vra = -27.114, -25.56  # vafrom - vato
ratio = 1.0
b = 1.0865

# Measured power flows
pr = 2.46
qr = -0.308

r, x = calculate_gkm_bkm(pr, qr, vsm, vrm, vsa, vra, ratio, b)

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
