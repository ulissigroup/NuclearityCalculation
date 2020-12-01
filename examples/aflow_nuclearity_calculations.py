from dask_kubernetes import KubeCluster
import dask.bag as db 
from joblib import Memory
from dask.distributed import Client
import functools
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic,get_initial_aflow_results
from surface_nuclearity_calculator import slab_enumeration,surface_nuclearity_calculator
#from dask.cache import Cache
from dask.distributed import progress
from pymatgen.io.ase import AseAtomsAdaptor

# Set the up a kube dask cluster
cluster = KubeCluster.from_yaml('worker-spec.yml')

# Adapt seems to be having problems, used fixed scaling
cluster.scale(3)
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
    nuclearity=[b.compound,b.auid,slab.miller_index,slab.shift,nuclearity_result[0],nuclearity_result[1],x_active]
    
    return {'slab':slab,
            'slab_atoms':slab_atoms,
            'nuclearity':nuclearity}

location = './cachedir'
memory = Memory(location,verbose=1)
get_initial_aflow_results = memory.cache(get_initial_aflow_results)
actives = set(['Pd', 'Pt', 'Rh', 'Ru', 'Ag', 'Ir'])
hosts = set(['Zn', 'Cd', 'Ga', 'Al', 'In'])

print("getting intital results")
all_aflow_binaries = get_initial_aflow_results(enthalpy_formation_atom=-0.1)
print("Total aflow binaries found = ",len(all_aflow_binaries))
active_inactive_aflow_binaries = list(filter(lambda r: select_bimetallic(r,actives,hosts), all_aflow_binaries))[:1]
print("Number of bimetallics found = ",len(active_inactive_aflow_binaries))

active_inactive_aflow_binaries_bag = db.from_sequence(active_inactive_aflow_binaries)
print("structure generation")
all_structures = active_inactive_aflow_binaries_bag.map(lambda b: AseAtomsAdaptor.get_structure(b.atoms(pattern='CONTCAR.relax*', quippy=False, keywords=None, calculator=None)))
print("slab enumeration")
all_slabs_list = all_structures.map(slab_enumeration, active_inactive_aflow_binaries_bag).persist()
all_slabs_list = all_slabs_list.flatten().persist()
print("nuclearity calculations")
nuclearity_results = all_slabs_list.map(slab_nuclearity,actives).persist()

progress(nuclearity_results)
nuclearity_results.compute()
