from dask_kubernetes import KubeCluster
import dask.bag as db 
from joblib import Memory
from dask.distributed import Client
import functools
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic,get_initial_aflow_results
from surface_nuclearity_calculator import slab_enumeration,surface_nuclearity_calculator
from dask.cache import Cache
from dask.distributed import progress
from pymatgen.io.ase import AseAtomsAdaptor

cache = Cache(2e9)  # Leverage two gigabytes of memory
cache.register()    # Turn cache on globally

# Set the up a kube dask cluster
cluster = KubeCluster.from_yaml('worker-spec.yml', scheduler='remote')
# cluster.adapt(minimum=0,maximum=10)
cluster.scale(20)
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
    [b,structure,slab] = master
    for i in range(0,len(b.species)):
        if b.species[i] in actives:
            x_active = b.stoich[i]
    unitCell_atoms = AseAtomsAdaptor.get_atoms(slab)
    nuclearity_result = surface_nuclearity_calculator(unitCell_atoms,structure,list(actives))
    slab_atoms=unitCell_atoms
    nuclearity=[b.compound,b.auid,slab.miller_index,slab.shift,nuclearity_result[0],nuclearity_result[1],x_active]
    return [slab,slab_atoms,nuclearity]

location = './cachedir'
memory = Memory(location,verbose=1)
get_initial_aflow_results = memory.cache(get_initial_aflow_results)
actives = set(['Pd', 'Pt', 'Rh', 'Ru', 'Ag', 'Ir'])
hosts = set(['Zn', 'Cd', 'Ga', 'Al', 'In'])

print("getting intital results")
all_aflow_binaries = get_initial_aflow_results(enthalpy_formation_atom=-0.1)
print("Total aflow binaries found = ",len(all_aflow_binaries))
active_inactive_aflow_binaries = list(filter(lambda r: select_bimetallic(r,actives,hosts), all_aflow_binaries))
print("Number of bimetallics found = ",len(active_inactive_aflow_binaries))
active_inactive_aflow_binaries_bag = db.from_sequence(active_inactive_aflow_binaries[0:2])
print("structure generation")
all_structures = active_inactive_aflow_binaries_bag.map(lambda b: AseAtomsAdaptor.get_structure(b.atoms(pattern='CONTCAR.relax*', quippy=False, keywords=None, calculator=None))).compute()
print("slab enumeration")
all_slabs_list = db.from_sequence(all_structures).map(lambda struc: slab_enumeration(struc)).compute()
print("masterlist assignment")
masterlist=[]
for i in range(0,len(all_structures)):
    for slab in all_slabs_list[i]:
        masterlist.append([active_inactive_aflow_binaries[i],all_structures[i],slab])
print("nuclearity calculations")
nuclearity_results = db.from_sequence(masterlist).map(lambda master: slab_nuclearity(master,actives)).compute()
