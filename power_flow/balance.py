from numpy import c_, zeros, pi, ones, exp, argmax, conj
from numpy import flatnonzero as find
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.loadcase import loadcase
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS


def calc_mis(casedata):
    ppc = loadcase(casedata)

    ## add zero columns to branch for flows if needed
    if ppc["branch"].shape[1] < QT:
        ppc["branch"] = c_[ppc["branch"],
        zeros((ppc["branch"].shape[0],
               QT - ppc["branch"].shape[1] + 1))]

    ## convert to internal indexing
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
    vcb = ones(V0.shape)  # create mask of voltage-controlled buses
    vcb[pq] = 0  # exclude PQ buses
    k = find(vcb[gbus])  # in-service gens at v-c buses
    V0[gbus[k]] = gen[on[k], VG] / abs(V0[gbus[k]]) * V0[gbus[k]]

    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Sbus = makeSbus(baseMVA, bus, gen)
    mis = V0 * conj(Ybus * V0) - Sbus
