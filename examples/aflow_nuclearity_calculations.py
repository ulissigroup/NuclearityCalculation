from dask_kubernetes import KubeCluster
import dask.bag as db
from joblib import Memory
from dask.distributed import Client
import functools
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic,get_initial_aflow_results, aflow_object_to_atoms
from surface_nuclearity_calculator import slab_enumeration,surface_nuclearity_calculator
from dask.distributed import progress
from pymatgen.io.ase import AseAtomsAdaptor
import dask

# Set the up a kube dask cluster
cluster = KubeCluster.from_yaml('worker-spec.yml')

# Adapt seems to be having problems, used fixed scaling
cluster.scale(40)
client = Client(cluster)

### Code to upload surface_nuclearity code to every worker as they start/restart
# https://stackoverflow.com/questions/57118226/how-to-properly-use-dasks-upload-file-to-pass-local-code-to-workers
fname = 'surface_nuclearity_calculator.py'
with open(fname, 'rb') as f:
  data = f.read()

def _worker_upload(dask_worker, *, data, fname):
  dask_worker.loop.add_callback(
    callback=dask_worker.upload_file,
    comm=None,  # not used
    filename=fname,
    data=data,
    load=True)

client.register_worker_callbacks(
  setup=functools.partial(
    _worker_upload, data=data, fname=fname,
  )
)


def slab_nuclearity(master,actives):
    b = master['active_inactive']
    structure = master['bulk_structure']
    slab = master['slab']

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

# Set up the cache directory - this will also be the local directory on each worker that will store cached results
location = './cachedir'
memory = Memory(location,verbose=1)

# Define the active and inactive elements to consider
actives = set(['Pd', 'Pt', 'Rh', 'Ru', 'Ag', 'Ir'])
hosts = set(['Zn', 'Cd', 'Ga', 'Al', 'In'])

# Gather the aflowlib bulk structures
get_initial_aflow_results = memory.cache(get_initial_aflow_results)
all_aflow_binaries = get_initial_aflow_results(enthalpy_formation_atom=-0.1)
print("Total aflow binaries found = ",len(all_aflow_binaries))

# Find all of the active/inactive combinations
active_inactive_aflow_binaries = list(filter(lambda r: select_bimetallic(r,actives,hosts), 
                                             all_aflow_binaries))[:700] # There is a problematic AlPd structure that stops this from going over 700
print("Number of active/inactive bimetallic structures found = ",len(active_inactive_aflow_binaries))

# Load all of the bulks into a dask bag, and get the atoms objects from aflowlib
active_inactive_aflow_binaries_bag = db.from_sequence(active_inactive_aflow_binaries, 
                                                      npartitions=len(active_inactive_aflow_binaries))
all_structures = active_inactive_aflow_binaries_bag.map(memory.cache(aflow_object_to_atoms))

# Enumerate all of the slabs, and repartition into 10k chunks of surfaces to work on
all_slabs_list = all_structures.map(memory.cache(slab_enumeration), 
                                    active_inactive_aflow_binaries_bag)
all_slabs_list = all_slabs_list.flatten().repartition(npartitions=10000)

# Run the nuclearity calculation on all of the slabs
nuclearity_results = all_slabs_list.map(memory.cache(slab_nuclearity),actives)

# Compute!
nuclearity_results = nuclearity_results.compute()