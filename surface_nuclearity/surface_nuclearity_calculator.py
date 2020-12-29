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

def slab_enumeration(bulk_structure, bulk):
    all_slabs = generate_all_slabs(bulk_structure,2,10,20,
                               bonds=None, tol=0.1, ftol=0.1, max_broken_bonds=0,
                               lll_reduce=False, center_slab=False, primitive=True,
                               max_normal_search=None, symmetrize=False, repair=False,
                               include_reconstructions=False, in_unit_planes=False)
    return [{'slab': slab,
             'bulk_structure': bulk_structure,
            'bulk': bulk} for slab in all_slabs]


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
    nn_analyzer = JmolNN()

    # Identify index of the surface atoms
    indices_list = []
    weights = [site.species.weight for site in struct]
    center_of_mass = np.average(struct.frac_coords,
                                weights=weights, axis=0)

    for idx, site in enumerate(struct):
        if site.frac_coords[2] > center_of_mass[2]:
            try:
                cn = nn_analyzer.get_cn(struct, idx, use_weights=True)
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
    neighborList = neighborlist.NeighborList(cutOff,
                                             self_interaction=False,
                                             bothways=True)
    neighborList.update(slab_atoms)
    connectivity_matrix = neighborList.get_connectivity_matrix()

    #Ignore connectivity with atoms which are not active or on the surface
    active_connectivity_matrix = connectivity_matrix.copy()
    active_list = [atom.symbol in actives and atom.index in surface_indices for atom in slab_atoms]

    if sum(active_list) == 0:
        # No active surface atoms!
        return {'max_nuclearity': 0,
                'nuclearities': []}

    else:
        active_connectivity_matrix = active_connectivity_matrix[active_list, :]
        active_connectivity_matrix = active_connectivity_matrix[:, active_list]

        # Make a graph-tool graph from the adjacency matrix
        graph = gt.Graph(directed=False)

        for i in range(active_connectivity_matrix.shape[0]):
            graph.add_vertex()

        graph.add_edge_list(np.transpose(active_connectivity_matrix.nonzero()))

        labels, hist = topology.label_components(graph, directed=False)

        return {'max_nuclearity': np.max(hist),
                'nuclearities': hist}


def surface_nuclearity_calculator(unitCell_atoms,bulk_structure,actives):
    #Check surface nuclearity for given slab and a repeated slab
    #Identify infinite or semifinite nuclearity cases
    replicated_atoms = unitCell_atoms.repeat((2,2,1))
    replicated_nuclearities = get_nuclearity_from_atoms(replicated_atoms,bulk_structure,actives)
    base_nuclearities = get_nuclearity_from_atoms(unitCell_atoms,bulk_structure,actives)

    if replicated_nuclearities['max_nuclearity'] == base_nuclearities['max_nuclearity']:
        return {'nuclearity': replicated_nuclearities['max_nuclearity'],
                'nuclearities': base_nuclearities['nuclearities']}
    elif replicated_nuclearities['max_nuclearity'] == 2*base_nuclearities['max_nuclearity']:
        return {'nuclearity': 'semi-finite',
                'nuclearities': base_nuclearities['nuclearities']}
    elif replicated_nuclearities['max_nuclearity'] == 4*base_nuclearities['max_nuclearity']:
        return {'nuclearity': 'infinite',
                'nuclearities': base_nuclearities['nuclearities']}
    else:
        return {'nuclearity':'somewhat-infinite',
                'base_nuclearities': base_nuclearities['nuclearities'],
                'replicated_nuclearities': replicated_nuclearities['nuclearities'],
                'nuclearities': base_nuclearities['nuclearities']}

def nuclearity_all_slabs_in_bulk(bulk,bulk_structure,actives):
  #slab enumeration
  if bulk_structure == None:
    bulk_structure = get_bulk_structure_from_entry(bulk)

  all_slabs_list = slab_enumeration(bulk_structure)

  #nuclearity calculation loop
  nuclearity_result = [surface_nuclearity_calculator(AseAtomsAdaptor.get_atoms(slab['slab']), bulk_structure,actives) for slab in all_slabs_list]
  return nuclearity_result

def get_bulk_structure_from_entry(bulk):
  #Assuming that the bulk belongs to Materials Project or Aflowlib
  try:
    with MPRester("MGOdX3P4nI18eKvE") as mpr:
      bulk_structure = mpr.get_structure_by_material_id(bulk['material_id'],
                                                        final=True, 
                                                        conventional_unit_cell
                                                        =True)
  except:
    bulk_structure = AseAtomsAdaptor.get_structure(bulk.atoms(pattern='CONTCAR.relax*', quippy=False, keywords=None, calculator=None))
  return bulk_structure