#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install joblib')


# In[22]:


from dask_kubernetes import KubeCluster
from dask.distributed import Client

cluster = KubeCluster.from_yaml('worker-spec.yml')
# cluster.adapt(minimum=0, maximum=10)  # or dynamically scale based on current workload
cluster.scale(10)
client = Client(cluster)
client.wait_for_workers(10)
client.upload_file('surface_nuclearity_calculator.py')


# In[20]:


from aflow import search,K
from joblib import Memory

def get_initial_aflow_results(nspecies=2,enthalpy_formation_atom=-0.1):
    result = search().select(K.nspecies==nspecies).filter(K.enthalpy_formation_atom<enthalpy_formation_atom)
    return result

location = './cachedir'
memory = Memory(location,verbose=1)
get_initial_aflow_results = memory.cache(get_initial_aflow_results)
db.from_sequence = memory.cache(db.from_sequence)
db.map = memory.cache(db.map)
#db.filter = memory.cache(db.filter)
db.compute = memory.cache(db.compute)


# In[21]:


import pickle
with open('all_results.pkl','rb') as filehandle:
    results = pickle.load(filehandle)
with open('bimetallics.pkl','rb') as filehandle:
    bimetallics = pickle.load(filehandle)


# In[18]:


#from distributed import Client
#client = Client(n_workers=50)#("tcp://127.0.0.1:37865")


# In[19]:


import dask.bag as db 
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic
import time

actives = ['Pd', 'Pt', 'Rh', 'Ru', 'Ag', 'Ir','Pd\n', 'Pt\n', 'Rh\n', 'Ru\n', 'Ag\n', 'Ir\n']
hosts = ['Zn', 'Cd', 'Ga', 'Al', 'In','Zn\n', 'Cd\n', 'Ga\n', 'Al\n', 'In\n']

print("getting intital results")
start1= time.time()
all_aflow_binaries = get_initial_aflow_results(enthalpy_formation_atom=-0.1)
end1 = time.time()
start2 = time.time()
all_aflow_binaries_bag = db.from_sequence(all_aflow_binaries)
end2 = time.time()
print("applying filter")
start3 = time.time()
outputs = all_aflow_binaries_bag.filter(lambda r: select_bimetallic(r,actives,hosts)).map(lambda b: bulk_nuclearity(b,actives)).compute()#.take(2)#joblib later
end3 = time.time()
print("nuclearity calculation")
#outputs=active_inactive_bag.map(lambda b: bulk_nuclearity(b,actives)).compute()
#outputs=db.from_sequence(bimetallics).map(lambda b: bulk_nuclearity(b,actives)).compute()
print(end1-start1, end2-start2, end3-start3)
#client.shutdown()


# In[ ]:


all_aflow_binaries_bag = db.from_sequence(all_aflow_binaries)


# In[24]:


import dask.bag as db 
from surface_nuclearity_calculator import bulk_nuclearity, select_bimetallic
import time
sample_bag=db.from_sequence([results[0], bimetallics[0]])


# In[ ]:


import time

start = time.time()
#active_inactive_bag = sample_bag.filter(lambda r: select_bimetallic(r,actives,hosts))
outputs=sample_bag.filter(lambda r: select_bimetallic(r,actives,hosts)).map(lambda b: bulk_nuclearity(b,actives)).compute()
end = time.time()
print(end-start)


# In[ ]:




