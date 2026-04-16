import functools
import math
from pathlib import Path
import warnings

import cotengra as ctg
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import xyzpy as xyz
from filelock import FileLock
from quimb.tensor import PEPS

try:
    from quimb.tensor.tnag.core import TensorNetworkGenVector
except ImportError:
    from quimb.tensor.tensor_arbgeom import TensorNetworkGenVector


heis_square_obc_energies_exact = {
    # Transverse field Ising model
    ("tfim_hz-3", "square_obc", 4): -3.13620,
    ("tfim_hz-3", "square_obc", 6): -3.15481,
    ("tfim_hz-3", "square_obc", 8): -3.16432,
    ("tfim_hz-3", "square_obc", 10): -3.17001,
    ("tfim_hz-3", "square_pbc", 4): -3.21557,
    ("tfim_hz-3", "square_pbc", 6): -3.20133,
    ("tfim_hz-3", "square_pbc", 8): -3.19751,
    ("tfim_hz-3", "square_pbc", 10): -3.19628,
    ("tfim_hz-5", "cubic_obc", 4): -5.12014,
    ("tfim_hz-5", "cubic_obc", 6): -5.13553,
    ("tfim_hz-5", "cubic_obc", 8): -5.14315,
    ("tfim_hz-5", "cubic_obc", 10): -5.147871,
    ("tfim_hz-5", "cubic_pbc", 4): -5.17914,
    ("tfim_hz-5", "cubic_pbc", 6): -5.17170,
    ("tfim_hz-5", "cubic_pbc", 8): -5.17070,
    ("tfim_hz-5", "cubic_pbc", 10): -5.170442,
    # Heisenberg model
    ("heis", "square_obc", 4): -0.5743254416,
    ("heis", "square_obc", 6): -0.603509,
    ("heis", "square_obc", 8): -0.619046,
    ("heis", "square_obc", 10): -0.628673,
    ("heis", "square_obc", 12): -0.635211,
    ("heis", "square_obc", 14): -0.639892,
    ("heis", "square_obc", 16): -0.643546,
    ("heis", "square_obc", 24): -0.652031,
    ("heis", "square_obc", 32): -0.656321,
    ("heis", "square_obc", 48): -0.6606672,
    ("heis", "square_obc", 64): -0.6628532,
    ("heis", "square_obc", 96): -0.6650443,
    ("heis", "square_obc", 128): -0.6661419,
    ("heis", "square_pbc", 4): -0.7017802005,
    ("heis", "square_pbc", 6): -0.678913,
    ("heis", "square_pbc", 8): -0.673468,
    ("heis", "square_pbc", 10): -0.671487,
    ("heis", "square_pbc", 12): -0.670722,
    ("heis", "square_pbc", 14): -0.670238,
    ("heis", "square_pbc", 16): -0.669935,
    ("heis", "square_pbc", 24): -0.669604,
    ("heis", "square_pbc", 32): -0.669515,
    ("heis", "square_pbc", 48): -0.6694614,
    ("heis", "square_pbc", 64): -0.6694443,
    ("heis", "square_pbc", 96): -0.6694485,
    ("heis", "square_pbc", 128): -0.6694409,
    ("heis", "cubic_obc", 3): -0.6801120427,
    ("heis", "cubic_obc", 4): -0.733465,
    ("heis", "cubic_obc", 6): -0.787000,
    ("heis", "cubic_obc", 8): -0.815166,
    ("heis", "cubic_obc", 10): -0.832311,
    ("heis", "cubic_obc", 12): -0.843845,
    ("heis", "cubic_pbc", 4): -0.915919,
    ("heis", "cubic_pbc", 6): -0.904842,
    ("heis", "cubic_pbc", 8): -0.903123,
    ("heis", "cubic_pbc", 10): -0.902633,
    ("heis", "cubic_pbc", 12): -0.902493,
    # Fermi-Hubbard model, half filling
    # U=8
    ("fermi_hubbard_U8", "square_obc", 4): -0.4255259042,
    ("fermi_hubbard_U8", "square_pbc", 4): -0.52975,
    ("fermi_hubbard_U8", "square_pbc", 6): -0.52778,
    ("fermi_hubbard_U8", "square_pbc", 8): -0.52625,
    ("fermi_hubbard_U8", "square_pbc", 10): -0.52540,
    ("fermi_hubbard_U8", "square_pbc", 12): -0.52458,
    ("fermi_hubbard_U8", "square_pbc", 14): -0.52474,
    ("fermi_hubbard_U8", "square_pbc", 16): -0.52434,
    # U=6
    ("fermi_hubbard_U6", "square_pbc", 4): -0.65881,
    ("fermi_hubbard_U6", "square_pbc", 6): -0.65944,
    ("fermi_hubbard_U6", "square_pbc", 8): -0.65875,
    ("fermi_hubbard_U6", "square_pbc", 10): -0.65800,
    ("fermi_hubbard_U6", "square_pbc", 12): -0.65736,
    # U=4
    ("fermi_hubbard_U4", "square_pbc", 4): -0.85100,
    ("fermi_hubbard_U4", "square_pbc", 6): -0.85736,
    ("fermi_hubbard_U4", "square_pbc", 8): -0.86016,
    ("fermi_hubbard_U4", "square_pbc", 10): -0.86120,
    ("fermi_hubbard_U4", "square_pbc", 12): -0.86076,
    # U=2
    ("fermi_hubbard_U2", "square_pbc", 4): -1.12650,
    ("fermi_hubbard_U2", "square_pbc", 6): -1.15158,
    ("fermi_hubbard_U2", "square_pbc", 8): -1.163594,
    ("fermi_hubbard_U2", "square_pbc", 10): -1.169080,
    ("fermi_hubbard_U2", "square_pbc", 12): -1.171868,
}


def get_energy_reference(model, geom, L):
    """Get the exact energy reference for a given model, geometry, and system
    size.
    """
    return heis_square_obc_energies_exact.get((model, geom, L), None)


def check_symmray_tensor_network(tn):
    """Verify symmray data are compatible with respect to network structure."""
    for x in tn.arrays:
        x.check()
    for ix, tids in tn.ind_map.items():
        if len(tids) == 1:
            continue
        ta, tb = tn._tids_get(*tids)
        axa = ta.inds.index(ix)
        axb = tb.inds.index(ix)
        ta.data.check_with(tb.data, (axa,), (axb,))


def draw_charges(tn, **draw_opts):
    """Draw the symmray charge distribution of a tensor network."""
    tn = tn.copy()
    ctags = set()
    for t in tn:
        ctag = f"Q{t.data.charge}"
        ctags.add(ctag)
        t.add_tag(ctag)
    tn.draw(ctags, **draw_opts)


def get_opt(
    progbar=False,
    **kwargs,
):
    """Get a contraction optimizer."""
    return ctg.ReusableHyperOptimizer(
        reconf_opts={},
        minimize="combo",
        max_time="rate:1e9",
        hash_method="b",
        optlib="cmaes",
        directory=True,
        progbar=progbar,
        **kwargs,
    )


def maybe_print(should_print: bool, msg: str, **kwargs):
    """Util for optionally printing base message and extra information."""
    if should_print:
        if kwargs:
            msg += " "
            msg += ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        print(msg)


def update_disk_dict(fname: str, new: dict):
    """Add new items to a dictionary saved on disk, or create a new one."""
    lockfile = Path(str(fname) + ".lock")
    with FileLock(lockfile):
        if not fname.exists():
            d = {}
        else:
            d = qu.load_from_disk(fname)
        d.update(new)
        qu.save_to_disk(d, fname)


def get_fname_psi(
    *,
    model: str,
    geom: str,
    L: int,
    dLy: int = 0,
    **kwargs,
):
    """Get disk name of wavefunction."""

    if dLy == 0:
        Lpath = Path(f"L{L}")
    else:
        Lpath = Path(f"L{L}x{L + dLy}")

    d = Path("data") / Path(model) / Path(geom) / Lpath
    d.mkdir(parents=True, exist_ok=True)

    option_spec = "".join(f"_{name}{value}" for name, value in kwargs.items())

    return d / Path(f"psi{option_spec}.dmp")


def get_fname_info(
    *,
    model: str,
    geom: str,
    L: int,
    **kwargs,
):
    """Get disk name of info dictionary."""
    d = Path("data") / Path(model) / Path(geom) / Path(f"L{L}")
    d.mkdir(parents=True, exist_ok=True)

    option_spec = "".join(f"_{name}{value}" for name, value in kwargs.items())

    return d / Path(f"info{option_spec}.dmp")


def get_info(
    *,
    model: str,
    geom: str,
    L: int,
    D: int,
    init: str,
    symm: str,
    schedule: str,
    seed: int,
    site_charge: str,
    **kwargs,
):
    """Load the information dictionary for a given wavefunction."""
    info_fname = get_fname_info(
        model=model,
        geom=geom,
        L=L,
        D=D,
        init=init,
        symm=symm,
        schedule=schedule,
        seed=seed,
        site_charge=site_charge,
        **kwargs,
    )
    if not info_fname.exists():
        return {}
    return qu.load_from_disk(info_fname)


def get_psi0_spin(
    *,
    geom: str,
    L: int,
    D: int,
    init: str,
    symm: str,
    seed: int = 0,
    subsizes: str = "maximal",
    site_charge: str = "checkerboard",
    dLy: int = 0,
    dLz: int = 0,
    noise_scale=1e-3,
    **kwargs,
):
    """Get an initial PEPS for the 2D Heisenberg model with open boundary
    conditions.

    Parameters
    ----------
    geom : str
        The geometry of the lattice. Options are:

        - 'square_obc': a square lattice with open boundary conditions.
        - 'square_pbc': a square lattice with periodic boundary conditions.
        - 'cubic_obc': a cubic lattice with open boundary conditions.
        - 'cubic_pbc': a cubic lattice with periodic boundary conditions.

    L : int
        The linear size of the main lattice side length.
    D : int
        The maximum bond dimension of the PEPS.
    init : str
        The type of initial state to prepare. Options are:

        - 'neel': a Neel state with alternating spins.
        - '0': all spins up.

    symm : str
        The symmetry to impose on the PEPS. Options are:

        - 'U1': the U(1) symmetry of total magnetisation.
        - 'Z2': the Z2 symmetry of total magnetisation.
        - 'None': no symmetry.

    seed : int, optional
        A random seed for initialising the PEPS.
    subsizes : str, optional
        The strategy for subsizes of the symmetry sectors to use.
    site_charge : str, optional
        The charge of the sites, for symmetry purposes.
    dLy : int, optional
        The difference between the x and y dimensions of the lattice.
    dLz : int, optional
        The difference between the x and z dimensions of the lattice.
    noise_scale : float, optional
        The scale of the background noise to add to sparse initial states.
    kwargs
        Additional keyword arguments to pass to the PEPS constructor.
    """
    lattice, cyclic = {
        "square_obc": ("square", False),
        "square_pbc": ("square", True),
        "cubic_obc": ("cubic", False),
        "cubic_pbc": ("cubic", True),
    }[geom]

    assert init in ("neel", "0", "1", "rand-uniform")

    Lx, Ly, Lz = L, L + dLy, L + dLz

    kwargs["Lx"] = Lx
    kwargs["Ly"] = Ly
    if lattice == "cubic":
        kwargs["Lz"] = Lz
    kwargs["cyclic"] = cyclic
    kwargs["bond_dim"] = D
    kwargs["seed"] = seed

    if init in ("neel", "0", "1"):
        # background noise
        kwargs["scale"] = noise_scale
    elif init == "rand-uniform":
        kwargs["dist"] = "uniform"
        kwargs["loc"] = -0.5
    else:
        raise ValueError(f"Unknown init: {init}")

    if (symm is None) or (symm == "None"):
        if lattice == "square":
            psi0 = qtn.PEPS.rand(**kwargs)
        else:  # lattice == "cubic"
            psi0 = qtn.PEPS3D.rand(**kwargs)

        if init in ("neel", "0", "1"):
            # impose neel order
            for site in psi0.gen_site_coos():
                data = psi0[site].data

                if (init == "0") or (
                    (init == "neel") and (sum(site) % 2 == 0)
                ):
                    selector = (0,) * data.ndim
                else:
                    # init == "1" or neel odd
                    selector = (0,) * (data.ndim - 1) + (1,)
                psi0[site].data[selector] = 1.0

        return psi0

    else:
        import symmray as sr

        if site_charge == "checkerboard":

            def site_charge(site):
                return sum(site) % 2

        elif site_charge == "uniform-0":

            def site_charge(site):
                return 0

        elif not callable(site_charge):
            raise ValueError(f"Unknown site_charge: {site_charge}")

        kwargs["symmetry"] = symm
        kwargs["subsizes"] = subsizes
        kwargs["site_charge"] = site_charge

        if lattice == "square":
            psi0 = sr.PEPS_abelian_rand(**kwargs)
        else:  # lattice == "cubic"
            psi0 = sr.PEPS3D_abelian_rand(**kwargs)

        if init in ("neel", "0"):
            # impose neel order
            for site in psi0.gen_site_coos():
                data = psi0[site].data
                if (init == "0") or (
                    (init == "neel") and (sum(site) % 2 == 0)
                ):
                    key = (0,) * data.ndim
                else:
                    # init == "1" or neel odd
                    key = (0,) * (data.ndim - 1) + (1,)
                selector = (0,) * data.ndim
                data.blocks[key][selector] = 1.0

        return psi0


def get_psi0_fermi_hubbard(
    *,
    geom: str,
    L: int,
    D: int,
    init: str,
    symm: str,
    seed: int = 0,
    subsizes: str = "maximal",
    site_charge: str = "uniform-1",
    dLy: int = 0,
    dLz: int = 0,
    **kwargs,
):
    """Get an initial PEPS for the 2D Fermi-Hubbard model with open boundary
    conditions.

    arameters
    ----------
    geom : str
        The geometry of the lattice. Options are:

        - 'square_obc': a square lattice with open boundary conditions.
        - 'square_pbc': a square lattice with periodic boundary conditions.
        - 'cubic_obc': a cubic lattice with open boundary conditions.
        - 'cubic_pbc': a cubic lattice with periodic boundary conditions.

    L : int
        The linear size of the main lattice side length.
    D : int
        The maximum bond dimension of the PEPS.
    """
    import symmray as sr

    lattice, cyclic = {
        "square_obc": ("square", False),
        "square_pbc": ("square", True),
        "cubic_obc": ("cubic", False),
        "cubic_pbc": ("cubic", True),
    }[geom]

    Lx, Ly = L, L + dLy

    kwargs["Lx"] = Lx
    kwargs["Ly"] = Ly
    kwargs["cyclic"] = cyclic
    kwargs["bond_dim"] = D
    kwargs["seed"] = seed
    kwargs["symmetry"] = symm
    kwargs["subsizes"] = subsizes

    if lattice == "cubic":
        Lz = L + dLz
        kwargs["Lz"] = Lz

    if init == "neel":
        # background noise
        kwargs["scale"] = 1e-3
    elif init == "rand-uniform":
        kwargs["dist"] = "uniform"
        kwargs["loc"] = -0.5
    else:
        raise ValueError(f"Unknown init: {init}")

    if site_charge == "checkerboard-2":
        # two electrons on every other site

        def site_charge(site):
            return 2 if sum(site) % 2 == 0 else 0

    elif site_charge == "uniform-0":
        # one electron on every site

        def site_charge(site):
            return 0

    elif site_charge == "uniform-1":
        # one electron on every site

        def site_charge(site):
            return 1

    elif site_charge == "checkerboard-U1U1":

        def site_charge(site):
            return (0, 1) if sum(site) % 2 == 0 else (1, 0)

    elif site_charge == "8th":
        if lattice == "square":
            sites = [(i, j) for i in range(Lx) for j in range(Ly)]
        else:
            sites = [
                (i, j, k)
                for i in range(Lx)
                for j in range(Ly)
                for k in range(Lz)
            ]

        empty_site_indices = distribute_items(len(sites), len(sites) // 8)
        empty_sites = set(sites[i] for i in empty_site_indices)

        def site_charge(site):
            if site in empty_sites:
                return 0
            else:
                return 1

    elif not callable(site_charge):
        raise ValueError(f"Unknown site_charge: {site_charge}")

    kwargs["site_charge"] = site_charge

    if symm == "Z2":
        kwargs["phys_dim"] = {0: 2, 1: 2}
    elif symm == "U1":
        kwargs["phys_dim"] = {0: 1, 1: 2, 2: 1}
    elif symm == "U1U1":
        kwargs["phys_dim"] = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    else:
        raise ValueError(f"Unknown symmetry: {symm}")

    if lattice == "square":
        psi0 = sr.PEPS_fermionic_rand(**kwargs)
    else:  # lattice == "cubic"
        psi0 = sr.PEPS3D_fermionic_rand(**kwargs)

    if init == "neel":
        if symm in ("Z2", "U1"):
            for site in psi0.gen_site_coos():
                t = psi0[site]
                key = (0,) * (t.ndim - 1) + (1,)
                if sum(site) % 2 == 0:
                    selector = (0,) * t.ndim
                else:
                    selector = (0,) * (t.ndim - 1) + (1,)
                try:
                    t.data.blocks[key][selector] = 1.0
                except KeyError:
                    warnings.warn("KeyError in neel order")
        elif symm == "U1U1":
            pass
        else:
            raise ValueError(f"Unknown symmetry: {symm}")

    return psi0


def get_psi0(
    *,
    model: str,
    geom: str,
    L: int,
    D: int,
    init: str,
    symm: str,
    seed: int,
    site_charge: str,
    **kwargs,
):
    """Get an initial wavefunction for a given model and geometry."""
    if ("heis" in model) or ("tfim" in model):
        return get_psi0_spin(
            geom=geom,
            L=L,
            D=D,
            init=init,
            symm=symm,
            seed=seed,
            site_charge=site_charge,
            **kwargs,
        )
    elif "fermi_hubbard" in model:
        return get_psi0_fermi_hubbard(
            geom=geom,
            L=L,
            D=D,
            init=init,
            symm=symm,
            seed=seed,
            site_charge=site_charge,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model or geometry: {model}, {geom}")


def get_psi_gauges_terms_lazy(
    geom,
    L,
    D,
    p,
    get_gauges=True,
    backend="numpy",
):
    from autoray.lazy import Variable

    if "square" in geom:
        if geom == "square_obc":
            cyclic = False
        elif geom == "square_pbc":
            cyclic = True
        else:
            raise ValueError(f"Unknown geometry: {geom}")

        edges = qtn.edges_2d_square(L, L, cyclic=cyclic)
        psi = qtn.PEPS.from_fill_fn(
            lambda shape: Variable(shape, backend=backend),
            L,
            L,
            D,
            phys_dim=p,
            cyclic=cyclic,
        )
    elif "cubic" in geom:
        if geom == "cubic_obc":
            cyclic = False
        elif geom == "cubic_pbc":
            cyclic = True
        else:
            raise ValueError(f"Unknown geometry: {geom}")

        edges = qtn.edges_3d_cubic(L, L, L, cyclic=cyclic)
        psi = qtn.PEPS3D.from_fill_fn(
            lambda shape: Variable(shape, backend=backend),
            L,
            L,
            L,
            D,
            phys_dim=p,
            cyclic=cyclic,
        )
    else:
        raise ValueError(f"Unknown geometry: {geom}")

    res = {"psi": psi}

    if get_gauges:
        res["gauges"] = {}
        psi.gauge_all_simple_(1, gauges=res["gauges"])

    O2 = Variable((p, p, p, p), backend=backend)
    res["terms"] = {edge: O2 for edge in edges}

    return res


def get_ham_tfim(*, geom, L, jx=-1.0, hz=-3.0, symm=None, dLy=0, dLz=0):
    """Get the TFIM Hamiltonian on a lattice."""
    lattice, cyclic = {
        "square_obc": ("square", False),
        "square_pbc": ("square", True),
        "cubic_obc": ("cubic", False),
        "cubic_pbc": ("cubic", True),
    }[geom]

    if lattice == "square":
        edges = qtn.edges_2d_square(L, L + dLy, cyclic=cyclic)
    else:  # lattice == "cubic"
        edges = qtn.edges_3d_cubic(L, L + dLy, L + dLz, cyclic=cyclic)

    if (symm is None) or (symm == "None"):
        X, Z = (qu.pauli(s, dtype="float64") for s in "XZ")
        jXX = jx * (X & X)
        bZ = hz * Z
        H2 = {edge: jXX for edge in edges}
        H1 = {site: bZ for edge in edges for site in edge}
    else:
        from symmray.hamiltonians import ham_tfim_from_edges

        H2 = ham_tfim_from_edges(symm, edges, jx=jx, hz=hz)
        H1 = None

    return qtn.LocalHamGen(H2, H1)


def get_ham_heis(*, geom, L, symm=None, dLy=0, dLz=0):
    """Get the Heisenberg Hamiltonian on a lattice."""
    lattice, cyclic = {
        "square_obc": ("square", False),
        "square_pbc": ("square", True),
        "cubic_obc": ("cubic", False),
        "cubic_pbc": ("cubic", True),
    }[geom]

    if lattice == "square":
        edges = qtn.edges_2d_square(L, L + dLy, cyclic=cyclic)
    else:  # lattice == "cubic"
        edges = qtn.edges_3d_cubic(L, L + dLy, L + dLz, cyclic=cyclic)

    if (symm is None) or (symm == "None"):
        terms = {edge: qu.ham_heis(2) for edge in edges}
    else:
        from symmray.hamiltonians import ham_heisenberg_from_edges

        terms = ham_heisenberg_from_edges(symm, edges)

    return qtn.LocalHamGen(terms)


def distribute_items(N, k):
    return [round(i * (N - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]


def get_ham_fermi_hubbard(
    *,
    geom,
    L,
    symm="U1",
    cyclic=False,
    dLy=0,
    dLz=0,
    U=8.0,
    mu=0.0,
):
    """Get the 2D Fermi-Hubbard Hamiltonian for a square lattice."""
    from symmray.hamiltonians import ham_fermi_hubbard_from_edges

    lattice, cyclic = {
        "square_obc": ("square", False),
        "square_pbc": ("square", True),
        "cubic_obc": ("cubic", False),
        "cubic_pbc": ("cubic", True),
    }[geom]

    if lattice == "square":
        edges = qtn.edges_2d_square(L, L + dLy, cyclic=cyclic)
    else:  # lattice == "cubic"
        edges = qtn.edges_3d_cubic(L, L + dLy, L + dLz, cyclic=cyclic)

    return qtn.LocalHamGen(
        ham_fermi_hubbard_from_edges(symm, edges, U=U, mu=mu)
    )


def get_ham(*, model, geom, L, symm, dLy=0, dLz=0, **kwargs):
    """Get a Hamiltonian for a given model and geometry."""
    assert not kwargs

    if "tfim" in model:
        hz = float(model.split("_hz")[-1])

        return get_ham_tfim(geom=geom, L=L, symm=symm, hz=hz, dLy=dLy, dLz=dLz)

    if "heis" in model:
        return get_ham_heis(geom=geom, L=L, symm=symm, dLy=dLy, dLz=dLz)

    elif "fermi_hubbard" in model:
        # extract interaction strength from model name
        U = float(model.split("_U")[-1])

        return get_ham_fermi_hubbard(
            geom=geom, L=L, symm=symm, U=U, dLy=dLy, dLz=dLz
        )

    else:
        raise ValueError(f"Unknown model or geometry: {model}, {geom}")


def get_su_schedule(schedule="D", N=None, D=None):
    """Get a imaginary time step and number of steps schedule for simple
    update, including ordering strategy and second order reflection.
    """
    if schedule[:2] == "DR":
        if len(schedule) == 2:
            factor = 1.0
        else:
            factor = float(schedule[2:])
        taus = [factor * D**-1.5]
        steps = [round(qu.log2(N) * 16 / taus[0])]
        second_order_reflect = True
        ordering = "random_sequential"

    elif schedule[:1] == "D":
        if len(schedule) == 1:
            factor = 1.0
        else:
            factor = float(schedule[1:])
        taus = [factor * D**-1.5]
        steps = [round(qu.log2(N) * 16 / taus[0])]
        second_order_reflect = True
        ordering = "smallest_last"

    elif schedule[0] == "S":
        if len(schedule) == 1:
            taus = [0.2]
        else:
            taus = [float(schedule[1:])]
        steps = [round(qu.log2(N) * 16 / taus[0])]
        second_order_reflect = True
        ordering = "smallest_last"

    elif schedule == "A":
        tau_i = 2.0
        tau_f = 0.02
        step_i = 4 * round(np.log(N))
        step_f = 4 * round(np.log(N))
        n_steps = 6
        taus = np.geomspace(tau_i, tau_f, num=n_steps)
        steps = np.geomspace(step_i, step_f, num=n_steps, dtype=int)
        second_order_reflect = True
        ordering = None

    elif schedule == "B":
        tau_i = 0.5
        tau_f = 0.02
        step_i = 6 * round(np.log(N))
        step_f = 6 * round(np.log(N))
        n_steps = 6
        taus = np.geomspace(tau_i, tau_f, num=n_steps)
        steps = np.geomspace(step_i, step_f, num=n_steps, dtype=int)
        second_order_reflect = True
        ordering = None

    elif schedule[0] == "L":
        if len(schedule) == 1:
            taus = [0.2]
        else:
            taus = [float(schedule[1:])]
        steps = [round(qu.log2(N) * 16 / taus[0])]
        second_order_reflect = True
        ordering = "largest_first"

    elif schedule[0] == "R":
        if len(schedule) == 1:
            taus = [0.2]
        else:
            taus = [float(schedule[1:])]
        steps = [round(qu.log2(N) * 16 / taus[0])]
        second_order_reflect = True
        ordering = None

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return {
        "taus": taus,
        "steps": steps,
        "second_order_reflect": second_order_reflect,
        "ordering": ordering,
    }


def run_simple_update(
    psi0: TensorNetworkGenVector,
    ham,
    D=None,
    schedule="D",
    cutoff=1e-12,
    tol=1e-8,
    tau_modifier=1.0,
    plot_every=None,
    gauges=None,
    progbar=False,
    **su_opts,
) -> dict:
    """Run a simple update evolution of a given wavefunction and hamiltonian."""
    if D is None:
        D = psi0.max_bond()

    s = get_su_schedule(schedule=schedule, N=psi0.nsites, D=D)

    if gauges is not None:
        psi0 = (psi0, gauges)

    maybe_print(
        progbar, "Running simple update...", cutoff=cutoff, schedule=schedule
    )

    with xyz.Timer() as timer, xyz.MemoryMonitor() as mem:
        su = qtn.SimpleUpdateGen(
            psi0=psi0,
            ham=ham,
            D=D,
            second_order_reflect=s["second_order_reflect"],
            ordering=s["ordering"],
            gate_opts=dict(cutoff=cutoff),
            compute_energy_every=0,
            compute_energy_final=False,
            tol=tol,
            plot_every=plot_every,
            progbar=progbar,
            **su_opts,
        )

        for tau, step in zip(s["taus"], s["steps"]):
            su.evolve(tau=tau_modifier * tau, steps=step)

    return {
        "psi": su.get_state(
            absorb_gauges="return" if gauges is not None else True
        ),
        "time": timer.time,
        "memory": mem.peak,
        "sdiff": su.gauge_diffs[-1],
        "its": su.n,
        "su": su,
    }


def memoize_groundstate_simple_update(
    *,
    model,
    geom,
    L,
    D,
    init="neel",
    init_Dlim="neel",
    symm=None,
    schedule="D",
    seed=0,
    site_charge=None,
    save_to_disk=True,
    overwrite=False,
    overwrite_previous=False,
    cutoff=1e-12,
    tau_modifier=0.5,
    plot_every=None,
    progbar=False,
    **psi_extras,
) -> dict:
    """Get or compute a simple update wavefunction, optionally saving to disk.

    Parameters
    ----------
    model : {"heis", "fermi_hubbard_U{}"}, str
        The model to consider.
    geom : {"square_obc"}, str
        The geometry of the lattice.
    L : int
        The linear size of the lattice.
    D : int
        The maximum bond dimension of the PEPS.
    init : str, optional
        The type of initial state to prepare. Options are:

        - "neel": a Neel state with alternating spins.
        - "rand-uniform": a random state with uniform distribution.
        - "D-1": start with the wavefunction of bond dimension D-1.

    init_Dlim : str, optional
        If using `init="D-1"`, the initial state to prepare for bond dimension
        D=1.
    symm : str, optional
        The symmetry to impose on the PEPS.
    schedule : str, optional
        The schedule of imaginary time steps to use.
    seed : int, optional
        A random seed for initialising the PEPS
    site_charge : str, optional
        How to distribute charges on the sites.
    save_to_disk : bool, optional
        Whether to save the wavefunction to disk.
    overwrite : bool, optional
        Whether to re-generate and overwrite the wavefunction even if it
        already exists on disk.
    cutoff : float, optional
        The cutoff to use for the simple update gate compression.
    progbar : bool, optional
        Whether to show a progress bar.
    psi_extras
        Additional keyword arguments to pass to the PEPS constructor.
    """

    psi_opts = dict(
        model=model,
        geom=geom,
        L=L,
        D=D,
        init=init,
        symm=symm,
        schedule=schedule,
        seed=seed,
        site_charge=site_charge,
        **psi_extras,
    )

    psi_fname = get_fname_psi(**psi_opts)

    if not overwrite and psi_fname.exists():
        maybe_print(progbar, f"Loading su wavefunction from disk: {psi_fname}")
        psi = qu.load_from_disk(psi_fname)
        info = get_info(**psi_opts)
        info_psi = {
            "time_su": info.get("time_su", np.nan),
            "memory_su": info.get("memory_su", np.nan),
            "sdiff_su": info.get("sdiff_su", np.nan),
            "its_su": info.get("its_su", np.nan),
        }
    else:
        maybe_print(progbar, "Creating su wavefunction...", **psi_opts)

        if (overwrite == "improve") and psi_fname.exists():
            # restart optimization
            psi0 = qu.load_from_disk(psi_fname)

        elif init in ("D-1", "D", "D+1"):
            if init == "D-1":
                Dprev = D - 1
            elif init == "D":
                Dprev = D
            elif init == "D+1":
                Dprev = D + 1

            init0 = init_Dlim if Dprev == 1 else init

            psi0 = memoize_groundstate_simple_update(
                model=model,
                geom=geom,
                L=L,
                D=Dprev,
                init=init0,
                init_Dlim=init_Dlim,
                symm=symm,
                schedule=schedule,
                seed=seed,
                site_charge=site_charge,
                overwrite=overwrite_previous,
                plot_every=plot_every,
                progbar=progbar,
                **psi_extras,
            )["psi"]
        else:
            # get random initial state directly
            psi0 = get_psi0(
                model=model,
                geom=geom,
                L=L,
                D=D,
                init=init,
                symm=symm,
                seed=seed,
                site_charge=site_charge,
                **psi_extras,
            )

        ham = get_ham(model=model, geom=geom, L=L, symm=symm)
        result = run_simple_update(
            psi0,
            ham,
            D=D,
            schedule=schedule,
            cutoff=cutoff,
            tau_modifier=tau_modifier,
            plot_every=plot_every,
            progbar=progbar,
        )
        info_psi = {
            "time_su": result["time"],
            "memory_su": result["memory"],
            "sdiff_su": result["sdiff"],
            "its_su": result["its"],
        }
        if save_to_disk:
            qu.save_to_disk(result["psi"], psi_fname)
            info_fname = get_fname_info(**psi_opts)
            update_disk_dict(info_fname, info_psi)

        psi = result["psi"]

    return {"psi": psi, **info_psi}


def compute_energy_full_square_obc(
    psi: PEPS,
    ham,
    chi,
    cutoff=1e-12,
    mode="mps",
    single_layer=False,
    optimize=None,
    **kwargs,
) -> dict:
    """For the given wavefunction and Hamiltonian, compute the energy, via
    full boundary contraction.

    Parameters
    ----------
    psi : qtn.PEPS
        The wavefunction.
    ham : qtn.LocalHam2D
        The Hamiltonian.
    chi : int
        The maximum bond dimension to use.
    cutoff : float, optional
        The cutoff to use for the boundary contraction.
    mode : str, optional
        Boundary compression strategy.
    single_layer : bool, optional
        Whether to only contract a single layer of the PEPS.
    kwargs
        Additional keyword arguments to pass to the boundary contraction.
    """
    if single_layer:
        kwargs["layer_tags"] = None

    if optimize is None or optimize == "auto":
        optimize = get_opt()

    with xyz.Timer() as timer, xyz.MemoryMonitor() as mem:
        terms_h = {
            (cooa, coob): hab
            for (cooa, coob), hab in ham.terms.items()
            if cooa[0] == coob[0]
        }
        terms_v = {
            (cooa, coob): hab
            for (cooa, coob), hab in ham.terms.items()
            if cooa[1] == coob[1]
        }

        energy_h = (
            psi.compute_local_expectation(
                terms_h,
                max_bond=chi,
                cutoff=cutoff,
                normalized=True,
                mode=mode,
                contract_optimize=optimize,
                **kwargs,
            )
            / psi.nsites
        )
        energy_v = (
            psi.compute_local_expectation(
                terms_v,
                max_bond=chi,
                cutoff=cutoff,
                normalized=True,
                mode=mode,
                contract_optimize=optimize,
                **kwargs,
            )
            / psi.nsites
        )
        energy = energy_h + energy_v

    return {
        "energy": energy,
        "time": timer.time,
        "memory": mem.peak,
    }


def memoize_energy_full(
    *,
    model,
    geom,
    L,
    D,
    init,
    symm,
    seed,
    site_charge,
    chi,
    schedule="D",
    mode="mps",
    single_layer=False,
    overwrite=False,
    overwrite_psi=False,
    save_to_disk=True,
    progbar=False,
    **psi_extra,
) -> dict:
    psi_opts = dict(
        model=model,
        geom=geom,
        L=L,
        D=D,
        init=init,
        symm=symm,
        schedule=schedule,
        seed=seed,
        site_charge=site_charge,
        **psi_extra,
    )

    info_fname = get_fname_info(**psi_opts)

    maybe_print(progbar, "Getting su wavefunction...", **psi_opts)

    psi = memoize_groundstate_simple_update(
        overwrite=overwrite_psi, progbar=progbar, **psi_opts
    )["psi"]

    should_compute = (
        overwrite
        or (not info_fname.exists())
        or (
            ("energy", chi, mode, single_layer)
            not in qu.load_from_disk(info_fname)
        )
    )

    if should_compute:
        maybe_print(
            progbar,
            "Computing energy...",
            chi=chi,
            mode=mode,
            single_layer=single_layer,
        )

        ham = get_ham(model=model, geom=geom, L=L, symm=symm)
        result = compute_energy_full_square_obc(
            psi=psi, ham=ham, chi=chi, mode=mode, single_layer=single_layer
        )
        energy = result["energy"]
        time_energy = result["time"]
        memory_energy = result["memory"]

        if save_to_disk:
            info_fname = get_fname_info(**psi_opts)
            update_disk_dict(
                info_fname,
                {
                    ("energy", chi, mode, single_layer): energy,
                    ("time_energy", chi, mode, single_layer): time_energy,
                    ("memory_energy", chi, mode, single_layer): memory_energy,
                },
            )
    else:
        maybe_print(
            progbar,
            "Loading energy from disk...",
            chi=chi,
            mode=mode,
            single_layer=single_layer,
        )
        info = qu.load_from_disk(info_fname)
        energy = info["energy", chi, mode, single_layer]
        time_energy = info["time_energy", chi, mode, single_layer]
        memory_energy = info.get(
            ("time_energy", chi, mode, single_layer), np.nan
        )

    maybe_print(progbar, "Done!", energy=energy)

    return {
        "energy": energy,
        "time": time_energy,
        "memory": memory_energy,
    }


def compute_energy_cluster(
    psi: TensorNetworkGenVector,
    ham,
    max_size,
    grow_from="alldangle",
    max_iterations=256,
    tol=5e-6,
    damping=0.1,
    optimize=None,
    progbar=False,
):
    """Compute the energy of `psi` with respect to `ham` using the 'single
    cluster' approximation.
    """
    psi = psi.copy()
    result = {}

    maybe_print(
        progbar,
        "Gauging wavefunction...",
        max_iterations=max_iterations,
        tol=tol,
        damping=damping,
    )

    with xyz.Timer() as timer:
        gauges = {}
        psi.gauge_all_simple_(
            max_iterations,
            tol=tol,
            damping=damping,
            gauges=gauges,
            progbar=progbar,
        )
    result["time_gauge"] = timer.time

    maybe_print(
        progbar,
        "Computing energy with single clusters...",
        max_size=max_size,
        grow_from=grow_from,
    )

    if optimize is None or optimize in ("auto", "none", "None"):
        optimize = get_opt()

    with xyz.Timer() as timer, xyz.MemoryMonitor() as mem:
        energy = (
            psi.compute_local_expectation_cluster(
                ham.terms,
                gauges=gauges,
                max_distance=max_size,
                mode="loopunion",
                grow_from=grow_from,
                optimize=optimize,
                progbar=progbar,
            )
            / psi.nsites
        )

    result["energy"] = energy
    result["time_energy"] = timer.time
    result["memory_energy"] = mem.peak
    return result


def memoize_energy_cluster(
    *,
    model,
    geom,
    L,
    D,
    init,
    symm,
    seed,
    site_charge,
    schedule="D",
    max_size=4,
    grow_from="alldangle",
    tol=5e-6,
    damping=0.1,
    max_iterations=256,
    overwrite=False,
    overwrite_psi=False,
    save_to_disk=True,
    optimize=None,
    progbar=False,
    **psi_extra,
):
    psi_opts = dict(
        model=model,
        geom=geom,
        L=L,
        D=D,
        init=init,
        symm=symm,
        schedule=schedule,
        seed=seed,
        site_charge=site_charge,
        **psi_extra,
    )

    info_fname = get_fname_info(**psi_opts)

    maybe_print(progbar, "Getting su wavefunction...", **psi_opts)

    psi = memoize_groundstate_simple_update(
        overwrite=overwrite_psi, progbar=progbar, **psi_opts
    )["psi"]

    should_compute = (
        overwrite
        or (not info_fname.exists())
        or (
            ("energy_cluster", max_size, grow_from)
            not in qu.load_from_disk(info_fname)
        )
    )

    if should_compute:
        maybe_print(
            progbar,
            "Computing energy...",
            max_size=max_size,
            grow_from=grow_from,
        )

        ham = get_ham(model=model, geom=geom, L=L, symm=symm)
        result = compute_energy_cluster(
            psi=psi,
            ham=ham,
            max_size=max_size,
            grow_from=grow_from,
            tol=tol,
            damping=damping,
            max_iterations=max_iterations,
            optimize=optimize,
            progbar=progbar,
        )
        energy_cluster = result["energy"]
        time_gauge_cluster = result["time_gauge"]
        time_energy_cluster = result["time_energy"]
        memory_energy_cluster = result["memory_energy"]

        if save_to_disk:
            new_info = {
                (
                    "energy_cluster",
                    max_size,
                    grow_from,
                ): energy_cluster,
                (
                    "time_gauge_cluster",
                    max_size,
                    grow_from,
                ): time_gauge_cluster,
                (
                    "time_energy_cluster",
                    max_size,
                    grow_from,
                ): time_energy_cluster,
                (
                    "memory_energy_cluster",
                    max_size,
                    grow_from,
                ): memory_energy_cluster,
            }
            update_disk_dict(info_fname, new_info)
    else:
        maybe_print(
            progbar,
            "Loading energy from disk...",
            max_size=max_size,
            grow_from=grow_from,
        )
        info = qu.load_from_disk(info_fname)
        energy_cluster = info[("energy_cluster", max_size, grow_from)]
        time_gauge_cluster = info[("time_gauge_cluster", max_size, grow_from)]
        time_energy_cluster = info[
            ("time_energy_cluster", max_size, grow_from)
        ]
        memory_energy_cluster = info.get(
            ("memory_energy_cluster", max_size, grow_from), np.nan
        )

    maybe_print(progbar, "Done!", energy_cluster=energy_cluster)

    return {
        "energy_cluster": energy_cluster,
        "time_gauge_cluster": time_gauge_cluster,
        "time_energy_cluster": time_energy_cluster,
        "memory_energy_cluster": memory_energy_cluster,
    }


def gen_rectangles(Lx, Ly, Bx=2, By=2, cyclic=False):
    import itertools

    for Bx, By in sorted(itertools.permutations((Bx, By))):
        if cyclic:
            cxr = range(Lx)
            cyr = range(Ly)
        else:
            cxr = range(0, Lx - Bx + 1)
            cyr = range(0, Ly - By + 1)
        for i, j in itertools.product(cxr, cyr):
            r = []
            bxr = range(i, i + Bx)
            byr = range(j, j + By)
            for ib, jb in itertools.product(bxr, byr):
                r.append((ib % Lx, jb % Ly))
            yield tuple(r)


def gen_cuboids(Lx, Ly, Lz, Bx=2, By=2, Bz=2, cyclic=False):
    import itertools

    for Bx, By, Bz in sorted(itertools.permutations((Bx, By, Bz))):
        if cyclic:
            cxr = range(Lx)
            cyr = range(Ly)
            czr = range(Lz)
        else:
            cxr = range(0, Lx - Bx + 1)
            cyr = range(0, Ly - By + 1)
            czr = range(0, Lz - Bz + 1)
        for i, j, k in itertools.product(cxr, cyr, czr):
            r = []
            bxr = range(i, i + Bx)
            byr = range(j, j + By)
            bzr = range(k, k + Bz)
            for ib, jb, kb in itertools.product(bxr, byr, bzr):
                r.append((ib % Lx, jb % Ly, kb % Lz))
            yield tuple(r)


def calc_cost_full_square_obc(
    L,
    D,
    p,
    chi,
    optimize=None,
    mode="mps",
    **kwargs,
):
    from autoray.experimental.complexity_tracing import compute_cost

    r = get_psi_gauges_terms_lazy("square_obc", L, D, p, get_gauges=False)
    psi = r["psi"]
    terms = r["terms"]

    terms_h = {
        (cooa, coob): hab
        for (cooa, coob), hab in terms.items()
        if cooa[0] == coob[0]
    }
    terms_v = {
        (cooa, coob): hab
        for (cooa, coob), hab in terms.items()
        if cooa[1] == coob[1]
    }

    lz_h = (
        psi.compute_local_expectation(
            terms_h,
            max_bond=chi,
            cutoff=0.0,
            normalized=True,
            mode=mode,
            contract_optimize=optimize,
            **kwargs,
        )
        / psi.nsites
    )
    lz_v = (
        psi.compute_local_expectation(
            terms_v,
            max_bond=chi,
            cutoff=0.0,
            normalized=True,
            mode=mode,
            contract_optimize=optimize,
            **kwargs,
        )
        / psi.nsites
    )
    lz = lz_h + lz_v

    return {
        "cost": math.log10(compute_cost(lz)),
        "peak": math.log2(lz.history_peak_size()),
        "max": math.log2(lz.history_max_size()),
        "num_terms": len(terms),
    }


def calc_cost_cluster(
    geom,
    L,
    D,
    p,
    gloops,
    grow_from,
    single_term=False,
    optimize=None,
    parallel=False,
    progbar=False,
):
    from autoray.experimental.complexity_tracing import compute_cost

    r = get_psi_gauges_terms_lazy(geom, L, D, p)
    psi: TensorNetworkGenVector = r["psi"]
    gauges = r["gauges"]
    terms = r["terms"]

    if optimize is None or optimize == "none":
        optimize = get_opt(parallel=parallel)

    if single_term:
        mid = L // 2
        if "square" in geom:
            cooa, coob = (mid, mid), (mid, mid + 1)
        elif "cubic" in geom:
            cooa, coob = (mid, mid, mid), (mid, mid, mid + 1)
        else:
            raise ValueError(f"Unknown geometry: {geom}")

        lz = psi.local_expectation_cluster(
            terms[cooa, coob],
            where=(cooa, coob),
            max_distance=gloops,
            mode="loopunion",
            gauges=gauges,
            grow_from=grow_from,
            optimize=optimize,
        )
    else:
        lz = psi.compute_local_expectation_cluster(
            terms,
            max_distance=gloops,
            mode="loopunion",
            gauges=gauges,
            grow_from=grow_from,
            optimize=optimize,
            progbar=progbar,
        )

    return {
        "cost": math.log10(compute_cost(lz)),
        "peak": math.log2(lz.history_peak_size()),
        "max": math.log2(lz.history_max_size()),
        "num_terms": len(terms),
    }


def calc_cost_gloop_expand(
    geom,
    L,
    D,
    p,
    gloops,
    grow_from,
    strict_size=False,
    single_term=False,
    optimize="random-greedy-128",
    parallel=False,
    info=None,
    progbar=False,
    **kwargs,
):
    from autoray.experimental.complexity_tracing import compute_cost

    r = get_psi_gauges_terms_lazy(geom, L, D, p)
    psi: TensorNetworkGenVector = r["psi"]
    gauges = r["gauges"]
    terms = r["terms"]

    if info is None:
        info = {}

    if optimize is None or optimize == "none":
        optimize = get_opt(parallel=parallel)

    gloops = parse_gloops(gloops, psi)

    if single_term:
        mid = L // 2 - 1
        if "square" in geom:
            cooa, coob = (mid, mid), (mid, mid + 1)
        elif "cubic" in geom:
            cooa, coob = (mid, mid, mid), (mid, mid, mid + 1)
        else:
            raise ValueError(f"Unknown geometry: {geom}")

        lz = psi.local_expectation_gloop_expand(
            terms[cooa, coob],
            where=(cooa, coob),
            gloops=gloops,
            gauges=gauges,
            grow_from=grow_from,
            strict_size=strict_size,
            optimize=optimize,
            info=info,
            progbar=progbar,
            **kwargs,
        )
    else:
        lz = psi.compute_local_expectation_gloop_expand(
            terms,
            gloops=gloops,
            gauges=gauges,
            grow_from=grow_from,
            strict_size=strict_size,
            optimize=optimize,
            info=info,
            progbar=progbar,
            **kwargs,
        )

    return {
        "cost": math.log10(compute_cost(lz)),
        "peak": math.log2(lz.history_peak_size()),
        "max": math.log2(lz.history_max_size()),
        "num_contractions": len(info["expecs"]),
        "num_terms": len(terms),
    }


def parse_gloops(max_size, psi):
    if max_size == 5.5:
        gloops = ()
        gloops += tuple(
            gen_cubes(
                psi.Lx,
                psi.Ly,
                psi.Lz,
                Bx=3,
                By=2,
                Bz=1,
                cyclic=psi.is_cyclic_x(),
            )
        )
    elif max_size in (7.4, 7.5):
        gloops = ()
        if max_size == 7.5:
            gloops += tuple(
                gen_cubes(
                    psi.Lx,
                    psi.Ly,
                    psi.Lz,
                    Bx=3,
                    By=2,
                    Bz=1,
                    cyclic=psi.is_cyclic_x(),
                )
            )
        gloops += tuple(
            gen_cubes(
                psi.Lx,
                psi.Ly,
                psi.Lz,
                Bx=2,
                By=2,
                Bz=2,
                cyclic=psi.is_cyclic_x(),
            )
        )
    elif max_size in (11.4,):
        gloops = ()
        gloops += tuple(
            gen_cubes(
                psi.Lx,
                psi.Ly,
                psi.Lz,
                Bx=3,
                By=2,
                Bz=2,
                cyclic=psi.is_cyclic_x(),
            )
        )
    else:
        gloops = max_size

    return gloops


def compute_energy_gloop_expand(
    psi: TensorNetworkGenVector,
    ham,
    max_size,
    grow_from="alldangle",
    mode="gloop_expand",
    max_iterations=256,
    tol=5e-6,
    damping=0.1,
    autocomplete=True,
    autoreduce=True,
    optimize="random-greedy-128",
    terms=None,
    info=None,
    progbar=False,
    **kwargs,
):
    psi = psi.copy()
    result = {}

    maybe_print(
        progbar,
        "Gauging wavefunction...",
        max_iterations=max_iterations,
        tol=tol,
        damping=damping,
    )

    with xyz.Timer() as timer:
        gauges = {}
        psi.gauge_all_simple_(
            max_iterations,
            tol=tol,
            damping=damping,
            gauges=gauges,
            progbar=progbar,
        )
    result["time_gauge"] = timer.time

    maybe_print(
        progbar,
        "Computing energy with gloop expansion...",
        max_size=max_size,
        grow_from=grow_from,
    )

    if info is None:
        info = {}

    if optimize is None or optimize == "none":
        optimize = get_opt()

    if terms is not None:
        terms = {term: ham.terms[term] for term in terms}
    else:
        terms = ham.terms

    if mode == "gloop_expand":
        # possibly lookup manual set of gloops
        gloops = parse_gloops(max_size, psi)
        num_joins = 1
        strict_size = False
    elif mode == "jgloop_expand":
        # form gloops by joining base loops only
        gloops = None
        num_joins = max_size
        strict_size = False
    elif mode == "rgloop_expand":
        # form gloops by generating restricted
        # join gloops and filtering by size
        num_joins = math.ceil((max_size - 2) / 2)
        gloops = tuple(psi.gen_gloops_sites(None, num_joins=num_joins))
        strict_size = max_size
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with xyz.Timer() as timer, xyz.MemoryMonitor() as mem:
        energy = (
            psi.compute_local_expectation_gloop_expand(
                terms,
                gauges=gauges,
                gloops=gloops,
                grow_from=grow_from,
                num_joins=num_joins,
                strict_size=strict_size,
                combine="prod",
                normalized=True,
                optimize=optimize,
                info=info,
                autocomplete=autocomplete,
                autoreduce=autoreduce,
                progbar=progbar,
                **kwargs,
            )
            / psi.nsites
        )

    result["energy"] = energy
    result["time_energy"] = timer.time
    result["memory_energy"] = mem.peak

    # info dict stores all the intermediate results, so these are ~free
    result["energy_norm_sum"] = (
        psi.compute_local_expectation_gloop_expand(
            terms,
            gauges=gauges,
            gloops=gloops,
            grow_from=grow_from,
            combine="sum",
            normalized="local",
            optimize=optimize,
            info=info,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            progbar=progbar,
        )
        / psi.nsites
    )
    result["energy_norm_sep"] = (
        psi.compute_local_expectation_gloop_expand(
            terms,
            gauges=gauges,
            gloops=gloops,
            grow_from=grow_from,
            combine="sum",
            normalized="separate",
            optimize=optimize,
            info=info,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            progbar=progbar,
        )
        / psi.nsites
    )

    return result


def memoize_energy_gloop_expand(
    *,
    model,
    geom,
    L,
    D,
    init,
    symm,
    seed,
    site_charge,
    schedule="D",
    max_size=4,
    mode="gloop_expand",
    grow_from="alldangle",
    tol=5e-6,
    damping=0.1,
    max_iterations=256,
    autocomplete=True,
    autoreduce=True,
    optimize="random-greedy-128",
    overwrite=False,
    overwrite_psi=False,
    save_to_disk=True,
    progbar=False,
    **psi_extra,
):
    psi_opts = dict(
        model=model,
        geom=geom,
        L=L,
        D=D,
        init=init,
        symm=symm,
        schedule=schedule,
        seed=seed,
        site_charge=site_charge,
        **psi_extra,
    )

    info_fname = get_fname_info(**psi_opts)

    maybe_print(progbar, "Getting su wavefunction...", **psi_opts)

    psi = memoize_groundstate_simple_update(
        overwrite=overwrite_psi, progbar=progbar, **psi_opts
    )["psi"]

    should_compute = (
        overwrite
        or (not info_fname.exists())
        or (
            (f"energy_{mode}", max_size, grow_from)
            not in qu.load_from_disk(info_fname)
        )
    )

    if should_compute:
        maybe_print(
            progbar,
            "Computing energy...",
            max_size=max_size,
            grow_from=grow_from,
        )

        ham = get_ham(model=model, geom=geom, L=L, symm=symm)
        result = compute_energy_gloop_expand(
            psi=psi,
            ham=ham,
            max_size=max_size,
            grow_from=grow_from,
            mode=mode,
            tol=tol,
            damping=damping,
            max_iterations=max_iterations,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            optimize=optimize,
            progbar=progbar,
        )
        energy_gloop_expand = result["energy"]
        energy_gloop_expand_norm_sum = result["energy_norm_sum"]
        energy_gloop_expand_norm_sep = result["energy_norm_sep"]
        time_gauge_gloop_expand = result["time_gauge"]
        time_energy_gloop_expand = result["time_energy"]
        memory_energy_gloop_expand = result["memory_energy"]

        if save_to_disk:
            new_info = {
                (
                    f"energy_{mode}",
                    max_size,
                    grow_from,
                ): energy_gloop_expand,
                (
                    f"energy_{mode}_norm_sum",
                    max_size,
                    grow_from,
                ): energy_gloop_expand_norm_sum,
                (
                    f"energy_{mode}_norm_sep",
                    max_size,
                    grow_from,
                ): energy_gloop_expand_norm_sep,
                (
                    f"time_gauge_{mode}",
                    max_size,
                    grow_from,
                ): time_gauge_gloop_expand,
                (
                    f"time_energy_{mode}",
                    max_size,
                    grow_from,
                ): time_energy_gloop_expand,
                (
                    f"memory_energy_{mode}",
                    max_size,
                    grow_from,
                ): memory_energy_gloop_expand,
            }
            update_disk_dict(info_fname, new_info)
    else:
        maybe_print(
            progbar,
            "Loading energy from disk...",
            max_size=max_size,
            grow_from=grow_from,
        )
        info = qu.load_from_disk(info_fname)
        energy_gloop_expand = info[(f"energy_{mode}", max_size, grow_from)]
        energy_gloop_expand_norm_sum = info[
            (f"energy_{mode}_norm_sum", max_size, grow_from)
        ]
        energy_gloop_expand_norm_sep = info[
            (f"energy_{mode}_norm_sep", max_size, grow_from)
        ]
        time_gauge_gloop_expand = info[
            (f"time_gauge_{mode}", max_size, grow_from)
        ]
        time_energy_gloop_expand = info[
            (f"time_energy_{mode}", max_size, grow_from)
        ]
        memory_energy_gloop_expand = info.get(
            (f"memory_energy_{mode}", max_size, grow_from), np.nan
        )

    maybe_print(progbar, "Done!", energy_gloop_expand=energy_gloop_expand)

    return {
        "energy_gloop_expand": energy_gloop_expand,
        "energy_gloop_expand_norm_sum": energy_gloop_expand_norm_sum,
        "energy_gloop_expand_norm_sep": energy_gloop_expand_norm_sep,
        "time_gauge_gloop_expand": time_gauge_gloop_expand,
        "time_energy_gloop_expand": time_energy_gloop_expand,
        "memory_energy_gloop_expand": memory_energy_gloop_expand,
    }


def gen_cubes(Lx, Ly, Lz, Bx=2, By=2, Bz=2, cyclic=False):
    if cyclic:
        cxrange = range(Lx)
        cyrange = range(Ly)
        czrange = range(Lz)
    else:
        cxrange = range(0, Lx - Bx + 1)
        cyrange = range(0, Ly - By + 1)
        czrange = range(0, Lz - Bz + 1)

    for i in cxrange:
        for j in cyrange:
            for k in czrange:
                r = []
                for ib in range(i, i + Bx):
                    for jb in range(j, j + By):
                        for kb in range(k, k + Bz):
                            r.append((ib % Lx, jb % Ly, kb % Lz))
                yield tuple(r)


def gen_strips(Lx, Ly, B, cyclic=False):
    if cyclic:
        cxrange = range(Lx)
        cyrange = range(Ly)
    else:
        cxrange = range(0, Lx - B + 1)
        cyrange = range(0, Ly - B + 1)

    for i in cxrange:
        r = []
        for ib in range(i, i + B):
            for jb in range(0, Ly):
                r.append((ib % Lx, jb % Ly))
        yield tuple(r)

    for j in cyrange:
        r = []
        for jb in range(j, j + B):
            for ib in range(0, Lx):
                r.append((ib % Lx, jb % Ly))
        yield tuple(r)


def gen_strips_diag(Lx, Ly, B, cyclic=False):
    def valid(i, j):
        return cyclic or (0 <= i < Lx and 0 <= j < Ly)

    if cyclic:
        cxrange = range(Lx)
        cyrange = range(Ly)
    else:
        cxrange = range(-Lx + 1, Ly - B)
        cyrange = range(-B + 2, Lx + Ly - B - 1)

    for c in cxrange:
        region = []
        for i in range(0, Lx):
            for jb in range(c + i, c + i + B + 1):
                if valid(i, jb):
                    region.append((i % Lx, jb % Ly))
        yield tuple(region)

    for c in cyrange:
        region = []
        for i in range(0, Lx):
            for jb in range(c - i, c - i + B + 1):
                if valid(i, jb):
                    region.append((i % Lx, jb % Ly))
        yield tuple(region)


def compute_wynn_eps(x, mode="wynn", atol=1e-12):
    """Compute the Wynn epsilon table for the given list of values."""
    if mode != "wynn":
        raise ValueError(f"Unknown mode: {mode}")

    N = len(x)
    eps = {-1: [0.0] * N, 0: list(x)}
    for r in range(N - 1):
        eps_rp1 = [None] * (N - r - 1)
        for n in range(N - r - 1):
            denom = eps[r][n + 1] - eps[r][n]
            if abs(denom) < atol:
                return eps

            eps_rp1[n] = eps[r - 1][n + 1] + 1 / denom

        eps[r + 1] = eps_rp1
    return eps


def estimate_wynn_with_error(x, mode="wynn4", atol=1e-12, plot=False):
    """Extrapolate the value of x using Wynn's epsilon algorithm and
    compute the error estimate.
    """
    x = [xi for xi in x if np.isfinite(xi)]
    N = len(x)

    if N < 1:
        return np.nan, np.nan

    if mode == "wynn":
        eps = compute_wynn_eps(x, mode=mode, atol=atol)
        kmax = max(k for k in eps if k % 2 == 0)
        last = eps[kmax]

        if kmax >= 2:
            kprev = kmax - 2
            prev = eps[kprev]
            est = last[-1]

            err_k = abs(est - prev[-1])
            if len(last) > 1:
                err_n = abs(est - last[-2])
                err = max(abs(est - last[-2]), abs(est - prev[-1]))
            else:
                err_n = abs(prev[-1] - prev[-2])

            err = max(err_n, err_k)

        else:
            # Not enough terms to extrapolate, use the last value as estimate
            est = last[-1]
            err = abs(est - last[-2])

        if N % 2 == 0:
            # only odd sequence lengths are valid, compare against ignoring first
            est1, err1 = estimate_wynn_with_error(x[1:], mode=mode, atol=atol)
            if err1 <= err:
                est, err = est1, err1

    elif mode == "wynn2":
        eps = compute_wynn_eps(x, mode="wynn", atol=atol)
        est = eps[2][-1]
        grad0 = abs(eps[0][-1] - eps[0][-2])
        grad2 = abs(eps[2][-1] - eps[2][-2])
        err = sum((grad0, grad2)) / 2

    elif mode == "wynn4":
        eps = compute_wynn_eps(x, mode="wynn", atol=atol)
        est = eps[4][-1]
        grad0 = abs(eps[0][-1] - eps[0][-2])
        grad2 = abs(eps[2][-1] - eps[2][-2])
        grad4 = abs(eps[4][-1] - eps[4][-2])
        err = sum((grad0, grad2, grad4)) / 3
        # err = (grad0 * grad2 * grad4) ** (1/3)
        # err = np.median((grad0, grad2, grad4))

    if plot:
        import xyzpy as xyz  # noqa
        import matplotlib as mpl  # noqa
        import matplotlib.pyplot as plt  # noqa

        mpl.style.use(xyz.get_neutral_style())

        _, ax = plt.subplots(figsize=(4, 3))

        color_extrap = (0.4, 0.8, 0.4)
        cm = xyz.cmoke(0.6, -0.2)
        cs = dict(zip(eps, np.linspace(0.0, 1.0, num=len(eps))))

        ax.axhspan(
            est - err / 2,
            est + err / 2,
            color=color_extrap,
            linestyle="--",
            linewidth=1,
            alpha=0.3,
        )
        ax.axhline(est, color=color_extrap, linestyle="--", linewidth=1)

        for k, yks in eps.items():
            if k % 2 == 0:
                ax.plot(range(k, k + len(yks)), yks, "x-", color=cm(cs[k]))

        ax.text(
            0.95,
            0.95,
            xyz.format_number_with_error(est, err),
            transform=ax.transAxes,
            color=(0.4, 0.8, 0.4),
            ha="right",
            va="top",
        )

        plt.show()
        plt.close()

    return est, err


def estimate_full_stats(
    chis,
    energies,
    mode="wynn4",
    atol=1e-15,
    plot=False,
):
    from scipy.optimize import curve_fit

    mask = np.isfinite(energies)
    x = chis[mask]
    y = energies[mask]
    logxinv = np.log10(1 / x)

    if mode == "weighted":
        # estimate final value as weighted mean
        yest = np.average(y[:], weights=x)
    elif isinstance(mode, int):
        # estimate final value as weighted mean of last `mode` values
        yest = np.average(y[-mode:], weights=x[-mode:])
    else:
        yest, _ = estimate_wynn_with_error(y, mode=mode, atol=atol)

    ydiffs = np.clip(np.abs(y - yest), atol, None)

    def f(x, a, b):
        return np.clip(a * x + b, np.log10(atol), None)

    gfit, _ = curve_fit(
        f,
        logxinv,
        np.log10(ydiffs),
    )

    # get model error at last accesible chi
    yerr = max(atol, 10 ** f(np.log10(1 / x[-1]), *gfit))

    if plot:
        import xyzpy as xyz  # noqa
        import matplotlib as mpl  # noqa
        import matplotlib.pyplot as plt  # noqa
        from matplotlib.ticker import FuncFormatter

        mpl.style.use(xyz.get_neutral_style())

        _, axs = plt.subplots(ncols=2, figsize=(8, 2))

        axs[0].plot(logxinv, np.log10(ydiffs), "x")
        xfit = np.linspace(max(logxinv), min(logxinv), 31)
        yfit = f(xfit, *gfit)

        axs[0].plot(xfit, yfit, "--")
        axs[0].invert_xaxis()
        axs[0].set_xlabel("$\\log_{10} 1/\\chi$")
        axs[0].set_title("Estimated errors", fontsize=10)

        axs[1].set_title("Estimated value", fontsize=10)
        axs[1].errorbar(x, y, 10 ** f(np.log10(1 / x), *gfit))
        axs[1].axhline(yest, color="green", linestyle="--")
        axs[1].set_xlabel("$\\chi$")
        axs[1].set_xscale("log", base=2)
        axs[1].xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f"{int(val):d}")
        )

        axs[1].text(
            0.95,
            0.95,
            xyz.format_number_with_error(yest, yerr),
            transform=axs[1].transAxes,
            color="green",
            ha="right",
            va="top",
        )

        plt.show()
        plt.close()

    return yest, yerr


def estimate_full_stats_into_ds(ds, **kwargs):
    import xarray as xr

    ds["energy_estimate"], ds["energy_estimate_error"] = xr.apply_ufunc(
        functools.partial(estimate_full_stats, **kwargs),
        ds["chi"],
        ds["energy"],
        input_core_dims=(["chi"], ["chi"]),
        output_core_dims=((), ()),
        vectorize=True,
    )


def estimate_gloop_stats(
    gloop_sizes,
    energies,
    mode="wynn4",
    mode_err="fit",
    atol=1e-15,
    plot=False,
):
    from scipy.optimize import curve_fit

    mask = np.isfinite(energies)

    if not np.any(mask):
        return np.nan, np.nan

    x = np.asarray(gloop_sizes)[mask]

    # gloop size 0 is really NN pairs
    x = np.clip(x, 2, None)
    y = np.asarray(energies)[mask]
    xinv = 1 / x

    if mode == "weighted":
        # estimate final value as weighted mean
        yest = np.average(y[:], weights=x)
    elif isinstance(mode, int):
        # estimate final value as weighted mean of last `mode` values
        yest = np.average(y[-mode:], weights=x[-mode:])
    else:
        yest, yerr = estimate_wynn_with_error(y, mode=mode)

    ydiffs = np.clip(np.abs(y - yest), atol, None)

    def f(x, a, b):
        return np.clip(a * x + b, np.log10(atol), None)

    gfit, _ = curve_fit(
        f,
        np.log10(xinv),
        np.log10(ydiffs),
    )

    if mode_err == "wynn":
        # already estimated with wynn
        if mode not in ("wynn", "shank", "wynn2", "wynn4"):
            # not valid for other modes
            yerr = np.nan
    elif mode_err == "fit":
        # get model error at last accesible gloop size
        yerr = max(atol, 10 ** f(np.log10(1 / x[-1]), *gfit))
    elif mode_err == "grad":
        yerr = abs(y[-1] - y[-2]) / abs(x[-1] - x[-2])
    else:
        raise ValueError(f"Unknown mode for error estimation: {mode_err}")

    if plot:
        import xyzpy as xyz  # noqa
        import matplotlib as mpl  # noqa
        import matplotlib.pyplot as plt  # noqa

        mpl.style.use(xyz.get_neutral_style())

        _, axs = plt.subplots(ncols=2, figsize=(8, 2))

        axs[0].plot(np.log10(xinv), np.log10(ydiffs), "x")
        xfit = np.log10(np.linspace(0.1, 0.5, 31))
        yfit = f(xfit, *gfit)
        axs[0].plot(xfit, yfit, "--")
        axs[0].invert_xaxis()
        axs[0].set_xlabel("$1/C$")
        # axs[0].set_xscale("log")
        # axs[0].set_yscale("log")
        axs[0].set_title("Estimated errors", fontsize=10)

        axs[1].set_title("Estimated value", fontsize=10)
        axs[1].errorbar(x, y, 10 ** f(np.log10(1 / x), *gfit))
        axs[1].axhspan(
            yest - yerr / 2,
            yest + yerr / 2,
            color=(0.4, 0.8, 0.4),
            linestyle="--",
            linewidth=1,
            alpha=0.3,
        )
        axs[1].axhline(
            yest, color=(0.4, 0.8, 0.4), linestyle="--", linewidth=1
        )
        axs[1].set_xlabel("$C$")

        axs[1].text(
            0.95,
            0.95,
            xyz.format_number_with_error(yest, yerr),
            transform=axs[1].transAxes,
            color=(0.4, 0.8, 0.4),
            ha="right",
            va="top",
        )

        plt.show()
        plt.close()

    return yest, yerr


def estimate_gloop_stats_into_ds(
    ds,
    y="energy_gloop_expand",
    ynew="energy_estimate",
    ynew_err="energy_estimate_error",
    mode="wynn4",
    mode_err="fit",
    atol=1e-15,
    plot=False,
    **kwargs,
):
    import xarray as xr

    kwargs["mode"] = mode
    kwargs["mode_err"] = mode_err
    kwargs["atol"] = atol
    kwargs["plot"] = plot

    ds[ynew], ds[ynew_err] = xr.apply_ufunc(
        functools.partial(estimate_gloop_stats, **kwargs),
        ds["max_size"],
        ds[y],
        input_core_dims=(["max_size"], ["max_size"]),
        output_core_dims=((), ()),
        vectorize=True,
    )


def estimate_gloop_wynn_into_ds(
    ds,
    y="energy_gloop_expand",
    mode="wynn4",
    atol=1e-15,
    plot=False,
):
    import xarray as xr

    ds["energy_wynn"], ds["energy_wynn_error"] = xr.apply_ufunc(
        functools.partial(
            estimate_wynn_with_error, mode=mode, atol=atol, plot=plot
        ),
        ds[y],
        input_core_dims=(["max_size"],),
        output_core_dims=((), ()),
        vectorize=True,
    )
