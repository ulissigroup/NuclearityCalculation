from dask_kubernetes import KubeCluster
import dask.bag as db 
from joblib import Memory
from dask.distributed import Client
import functools
from dask.cache import Cache
from dask.diagnostics import ProgressBar

cache = Cache(2e9)  # Leverage two gigabytes of memory
cache.register()    # Turn cache on globally

ProgressBar().register()

# Set the up a kube dask cluster
cluster = KubeCluster.from_yaml('worker-spec.yml', scheduler='remote')
# cluster.adapt(minimum=0,maximum=10)
cluster.scale(10)
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

from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic,get_initial_aflow_results 

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
active_inactive_aflow_binaries_bag = db.from_sequence(active_inactive_aflow_binaries)
print("nuclearity calculation")
nuclearity_results = active_inactive_aflow_binaries_bag.map(lambda b: bulk_nuclearity(b,actives)).compute()



