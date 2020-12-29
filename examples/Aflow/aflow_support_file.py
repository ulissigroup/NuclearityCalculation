import pickle
import glob
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import generate_all_slabs
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import scipy
from scipy.spatial.qhull import QhullError
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN, JmolNN
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from scipy import sparse
import numpy as np
import networkx as nx
from ase import Atoms
from aflow import search,K
import tqdm
from scipy.sparse.csgraph import connected_components
import graph_tool as gt
from graph_tool import topology
from surface_nuclearity_calculator import surface_nuclearity_calculator

def aflow_object_to_atoms(entry):
    return AseAtomsAdaptor.get_structure(entry.atoms(pattern='CONTCAR.relax*', quippy=False, keywords=None, calculator=None))

def get_initial_aflow_results(nspecies=2,enthalpy_formation_atom=-0.1):
    result = search(batch_size=10000
                   ).filter((K.nspecies==nspecies)&(K.enthalpy_formation_atom<enthalpy_formation_atom)
                           ).select(K.species,K.stoich,K.compound,K.auid)
    return list(tqdm.tqdm(result))

def slab_nuclearity(master,actives):
    try:
        b = master['bulk']
        structure = master['bulk_structure']
        slab = master['slab']
    except:
        b = master[0]['bulk']
        structure = master[0]['bulk_structure']
        slab = master[0]['slab']

    for i in range(0,len(b.species)):
        if b.species[i] in actives:
            x_active = b.stoich[i]

    unitCell_atoms = AseAtomsAdaptor.get_atoms(slab)
    nuclearity_result = surface_nuclearity_calculator(unitCell_atoms,structure,list(actives))
    slab_atoms=unitCell_atoms
    nuclearity = [b.compound,
                                b.auid,
                                slab.miller_index,
                                slab.shift,
                                nuclearity_result['nuclearity'],
                                nuclearity_result['nuclearities'],
                                x_active]
    return {'slab':slab,
            'slab_atoms':slab_atoms,
            'nuclearity':nuclearity}

def select_bimetallic(r,actives,hosts):
    # remove trailing \n from species
    species = [a.strip() for a in r.species]
    if len(actives.intersection(r.species))>0 and \
       len(hosts.intersection(r.species))>0:
        return True
    else:
        return False