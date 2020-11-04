from dask_kubernetes import KubeCluster
import dask.bag as db 
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic
from aflow import search,K
from joblib import Memory

cluster = KubeCluster.from_yaml('worker-spec.yml')
cluster.adapt(minimum=1, maximum=10)  # or dynamically scale based on current workload

def get_initial_aflow_results(nspecies=2,enthalpy_formation_atom=-0.1):
    result = search().select(K.nspecies==nspecies).filter(K.enthalpy_formation_atom<enthalpy_formation_atom)
    return result

location = './cachedir'
memory = Memory(location,verbose=1)
get_initial_aflow_results = memory.cache(get_initial_aflow_results)
actives = ['Pd', 'Pt', 'Rh', 'Ru', 'Ag', 'Ir','Pd\n', 'Pt\n', 'Rh\n', 'Ru\n', 'Ag\n', 'Ir\n']
hosts = ['Zn', 'Cd', 'Ga', 'Al', 'In','Zn\n', 'Cd\n', 'Ga\n', 'Al\n', 'In\n']

print("getting intital results")
all_aflow_binaries = get_initial_aflow_results(enthalpy_formation_atom=-3.1)
all_aflow_binaries_bag = db.from_sequence(all_aflow_binaries)
print("applying filter")
active_inactive_bag = all_aflow_binaries_bag.filter(lambda r: select_bimetallic(r,actives,hosts)>0).map(lambda b: bulk_nuclearity(b,actives)).compute()#.take(2)#joblib later
print("nuclearity calculation")
outputs=active_inactive_bag.map(lambda b: bulk_nuclearity(b,actives)).compute()
outputs=db.from_sequence(bimetallics).map(lambda b: bulk_nuclearity(b,actives)).compute()

