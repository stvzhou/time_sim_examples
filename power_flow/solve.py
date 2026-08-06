import numpy as np

from matpow import MatpowerCase
from validation import build_bus_energy_balances, validate_matpower_energy_balance
from pypower.api import runpf
from pypower.ppoption import ppoption


def validate_case(mpc: MatpowerCase):
    print("Max X: ", max([abs(b.br_x) for b in mpc.branch]))
    print("Min X: ", min([abs(b.br_x) for b in mpc.branch]))
    print("Max R: ", max([abs(b.br_r) for b in mpc.branch]))
    print("Min R: ", min([abs(b.br_r) for b in mpc.branch]))
    print("Max R/X: ", max([abs(b.br_r / b.br_x) for b in mpc.branch]))
    print("Out of service branches: ", len([b for b in mpc.branch if not b.br_status]))
    print("Out of service buses: ", len([b for b in mpc.bus if b.bus_type == 4]))
    print("In service generators: ", len([b for b in mpc.gen if b.gen_status]))
    print("Number of out of service generators: ", len([b for b in mpc.gen if not b.gen_status]))
    print("Number of slack bus: ", len([b for b in mpc.bus if b.bus_type == 3]))
    assert all(np.isfinite(b.br_x) for b in mpc.branch), "Branch x is invalid"
    assert all(np.isfinite(b.br_r) for b in mpc.branch), "Branch r is invalid"


mpc = MatpowerCase.from_mat("/usr/local/google/home/sxzhou/Downloads/test.mat")
# id_to_bus = build_bus_energy_balances(mpc)
# validate_matpower_energy_balance(
#     mpc, p_tol=0.0001, q_tol=0.0001
# )
# validate_case(mpc)
# print(id_to_bus[mpc.bus[20267].bus_i])

ppc = mpc.to_dict()

ppopt = ppoption(
    MODEL="AC",  # AC power flow model
    PF_ALG=1,  # 1 = Newton-Raphson ('NR')
    PF_TOL=1e-4,  # Convergence tolerance
    PF_MAX_IT=40,  # Maximum iteration limit
    ENFORCE_Q_LIMS=0,  # 0 = Do not enforce Q limits initially
    OUT_ALL=0,  # Do not print bus/branch/gen result tables
    VERBOSE=1,  # Print solver progress info
    CURRENT_BALANCE=0,
)
results, success = runpf(ppc, ppopt)
print(f"\nAC Power flow converged: {bool(success)}")
