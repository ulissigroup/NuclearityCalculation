# first line: 28
def get_initial_aflow_results(nspecies=2,enthalpy_formation_atom=-0.1):
    result = search(batch_size=10000
                   ).filter((K.nspecies==nspecies)&(K.enthalpy_formation_atom<enthalpy_formation_atom)
                           ).select(K.species,K.stoich,K.compound,K.auid)
    return list(tqdm.tqdm(result))
