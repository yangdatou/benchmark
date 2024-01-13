import os
import pyscf
from pyscf.lib import logger

def WaterCluster(n=64, basis="ccpvdz", verbose=4):
    xyz_path   = os.environ.get("XYZ_PATH", None)
    max_memory = os.environ.get("PYSCF_MAX_MEMORY", 4000)
    assert xyz_path is not None

    atom = ""
    with open(xyz_path, "r") as f:
        lines = f.readlines()
        atom = "".join(lines[(2):(2+3*n)])

    print(atom)

    mol = pyscf.gto.Mole()
    mol.atom  = atom
    mol.basis = basis
    mol.max_memory = int(max_memory)
    mol.verbose    = verbose
    mol.build()
    return mol

def setup_logger():
    log = pyscf.lib.logger.Logger(verbose=5)
    with open('/proc/cpuinfo') as f:
        for line in f:
            if 'model name' in line:
                log.note(line[:-1])
                break
    with open('/proc/meminfo') as f:
        log.note(f.readline()[:-1])
    log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))
    return log

def get_cpu_timings():
    t1 = logger.process_clock()
    t2 = logger.perf_counter()
    return t1, t2

def main():
    for n in [2, 4, 8, 16, 32, 48, 64]:
        m = WaterCluster(n=n, basis="ccpvdz", verbose=0)

        h = pyscf.scf.RHF(m)
        dm0 = h.get_init_guess(key="minao")

        t0 = get_cpu_timings()
        vj = h.get_jk(dm=dm0, hermi=1)[0]
        h.verbose = 5
        logger.timer(h, "n = %2d, get_veff" % n, *t0)

        assert h._eri is not None