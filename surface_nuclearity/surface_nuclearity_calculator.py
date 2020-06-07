'''
This submodule contains various functions that operate on 'ase.Atoms' 
objects and bulk structures from pymatgen.
'''

import scipy
from scipy.spatial.qhull import QhullError
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from ase import neighborlist
from ase.utils import natural_cutoffs
from scipy import sparse
import numpy as np
import networkx as nx
from ase import Atoms


def find_bulk_cn_dict(bulk_atoms):
    struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(struct)
    sym_struct = sga.get_symmetrized_structure()
    unique_indices = [equ[0] for equ in sym_struct.equivalent_indices]
    # Get a dictionary of unique coordination numbers
    # for atoms in each structure.
    # for example, Pt[1,1,1] would have cn=3 and cn=12
    # depends on the Pt atom.
    voronoi_nn = VoronoiNN()
    cn_dict = {}
    for idx in unique_indices:
        elem = sym_struct[idx].species_string
        if elem not in cn_dict.keys():
            cn_dict[elem] = []
        cn = voronoi_nn.get_cn(sym_struct, idx, use_weights=True)
        cn = float('%.5f' % (round(cn, 5)))
        if cn not in cn_dict[elem]:
            cn_dict[elem].append(cn)
    return cn_dict


def find_surface_atoms_indices(bulk_cn_dict, atoms):
    struct = AseAtomsAdaptor.get_structure(atoms, cls = None)
    voronoi_nn = VoronoiNN()
    # Identify index of the surface atoms
    indices_list = []
    weights = [site.species.weight for site in struct]
    center_of_mass = np.average(struct.frac_coords,
                                weights=weights, axis=0)
    for idx, site in enumerate(struct):
        if site.frac_coords[2] > center_of_mass[2]:
            try:
                cn = voronoi_nn.get_cn(struct, idx, use_weights=True)
                cn = float('%.5f' % (round(cn, 5)))
                # surface atoms are undercoordinated
                if cn < min(bulk_cn_dict[site.species_string]):
                    indices_list.append(idx)
            except RuntimeError:
                # or if pathological error is returned,
                # indicating a surface site
                indices_list.append(idx)
    return indices_list

def get_nuclearity_from_atoms(atoms,structure,actives):
    #Get surface nuclearity from given Atoms object
    slab_atoms = atoms.copy()
    #pick surface atoms
    bulk_atoms = AseAtomsAdaptor.get_atoms(structure)
    bulk_cn = find_bulk_cn_dict(bulk_atoms)
    surface_indices = find_surface_atoms_indices(bulk_cn, slab_atoms)

    #Generate connectivity matrix
    cutOff = natural_cutoffs(slab_atoms)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, 
                                             bothways=True)
    neighborList.update(slab_atoms)
    connectivity_matrix = neighborList.get_connectivity_matrix()    

    #Ignore connectivity with atoms which are not active or on the surface
    active_connectivity_matrix = connectivity_matrix.copy()
    for atom in slab_atoms:
        if atom.symbol not in actives:
            active_connectivity_matrix[atom.index,:] = 0
            active_connectivity_matrix[:,atom.index] = 0
        if atom.index not in surface_indices:
            active_connectivity_matrix[atom.index,:] = 0
            active_connectivity_matrix[:,atom.index] = 0            
    graph = nx.from_scipy_sparse_matrix(active_connectivity_matrix)

    #Remove host atoms which are showing up as single atom components
    lengths = []
    list1 = list(nx.connected_components(graph))
    list2 = list1.copy()
    for s in list1:
        for q in s:
            if slab_atoms[q].symbol not in actives:
                list2.remove(s)
                break
            if q not in surface_indices:
                list2.remove(s)
                break
            
    #Get list of nuclearities of all active sites on surface
    for l in list2:
        lengths.append(len(l))
    if len(lengths) == 0:
        max_nuclearity = 0
    else:
        max_nuclearity = max(lengths)
        
    return [max_nuclearity,lengths]

def surface_nuclearity_calculator(unitCell_atoms,bulk_structure,actives):
    #Check surface nuclearity for given slab and a repeated slab
    #Identify infinite or semifinite nuclearity cases
    slab_atoms = unitCell_atoms.repeat((2,2,1))
    slab_nuclearities = get_nuclearity_from_atoms(slab_atoms,bulk_structure,actives)
    unitCell_nuclearities = get_nuclearity_from_atoms(unitCell_atoms,bulk_structure,actives)
    if slab_nuclearities[0] == unitCell_nuclearities[0]:
        surface_nuclearity = slab_nuclearities
    elif slab_nuclearities[0] == 2*unitCell_nuclearities[0]:
        surface_nuclearity = ['semi-finite',slab_nuclearities[1]]
    elif slab_nuclearities[0] == 4*unitCell_nuclearities[0]:
        surface_nuclearity = ['infinite',slab_nuclearities[1]]
    else:
        surface_nuclearity = [{'unitCell': unitCell_nuclearities[0], 
                               'slab': slab_nuclearities[0]},
                              slab_nuclearities[1]]
    return (surface_nuclearity)
