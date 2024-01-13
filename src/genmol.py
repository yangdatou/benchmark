import pyscf
from pyscf import gto, scf, ao2mo
from pyscf.lib import logger

def WaterCluster(n=64, basis="ccpvdz", verbose=4):
    atom = ""
    with open("./h2o.txt", "r") as f:
        lines = f.readlines()
        atom = "".join(lines[:(3*n)])

    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.max_memory=4000
    mol.verbose = verbose
    mol.build()

    return mol

if __name__ == "__main__":
    for n in [2, 4, 8, 16, 32, 48, 64]:
        m = WaterCluster(n=n, basis="ccpvdz", verbose=0)
        m.max_memory = 16000

        h = scf.RHF(m)
        dm0 = h.get_init_guess(key="minao")

        t0 = (logger.process_clock(), logger.perf_counter())
        vj = h.get_jk(dm=dm0, hermi=1)[0]
        h.verbose = 5
        logger.timer(h, "n = %2d, get_veff" % n, *t0)

        assert h._eri is not None