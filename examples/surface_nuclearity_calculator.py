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
from pymatgen.analysis.local_env import VoronoiNN
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from scipy import sparse
import numpy as np
import networkx as nx
from ase import Atoms
from aflow import search,K
import tqdm


def get_initial_aflow_results(nspecies=2,enthalpy_formation_atom=-0.1):
    result = search(batch_size=10000
                   ).filter((K.nspecies==nspecies)&(K.enthalpy_formation_atom<enthalpy_formation_atom)
                           ).select(K.species)
    return list(tqdm.tqdm(result))

def bulk_nuclearity(b,actives):
    print("working...")
    bulk_atoms = b.atoms(pattern='CONTCAR.relax*', quippy=False, keywords=None, calculator=None)
    structure = AseAtomsAdaptor.get_structure(bulk_atoms)
    slab_list=slab_enumeration(structure)
    slab_atoms_list = []
    nuclearity_list = []
    for i in range(0,len(b.species)):
        if b.species[i] in actives:
            x_active = b.stoich[i]
    for slab in slab_list:
        unitCell_atoms = AseAtomsAdaptor.get_atoms(slab)
        nuclearity_result = surface_nuclearity_calculator(unitCell_atoms,structure,list(actives))
        slab_atoms_list.append(unitCell_atoms)
        nuclearity_list.append([b.compound,b.auid,slab.miller_index,slab.shift,nuclearity_result[0],nuclearity_result[1],x_active])
    return [slab_list,slab_atoms_list,nuclearity_list]

def select_bimetallic(r,actives,hosts):
    # remove trailing \n from species
    species = [a.strip() for a in r.species]
    if len(actives.intersection(r.species))>0 and \
       len(hosts.intersection(r.species))>0:
        return True
    else:
        return False

def slab_enumeration(bulk_structure):
    all_slabs = generate_all_slabs(bulk_structure,2,10,20,
                               bonds=None, tol=0.1, ftol=0.1, max_broken_bonds=0,
                               lll_reduce=False, center_slab=False, primitive=True,
                               max_normal_search=None, symmetrize=False, repair=False,
                               include_reconstructions=False, in_unit_planes=False)
    return all_slabs

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