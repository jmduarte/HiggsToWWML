import uproot
import pandas
import numpy as np
import pandas as pd
import h5py
import tables
import sys
filters = tables.Filters(complevel=7, complib='blosc')

path_sig = 'root://cmseos.fnal.gov//store/group/lpchbb/cmantill/pancakes/01/hadd/hww_mc-hadd/'

infiles = [path_sig+'RunIIAutumn18MiniAOD-102X_v15-v1_{}.root'.format(i) for i in range(50)]

outfile = 'data/raw/HWW.h5'
#entrystop = 100
entrystop = None

other_branches = ['run', 'luminosityBlock', 'event', 'MET_pt', 'MET_phi', 'nCustomAK8Puppi', 'nLHEPart']
gen_branches = ['LHEPart_pt', 'LHEPart_eta', 'LHEPart_phi', 'LHEPart_mass', 'LHEPart_pdgId']
jet_branches = ['CustomAK8Puppi_pt', 'CustomAK8Puppi_eta', 'CustomAK8Puppi_phi', 'CustomAK8Puppi_msoftdrop']

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)
    
def _transform(dataframe, max_particles=100, start=0, stop=-1):    
    return dataframe[dataframe.index.get_level_values(-1)<max_particles].unstack().fillna(0)

df_others = []
df_gens = []
df_jets = []
currententry = 0
for infile in infiles:
    print('converting {}'.format(infile))
    upfile = uproot.open(infile)
    tree = upfile['Events']
    
    df_other = tree.pandas.df(branches=other_branches, entrystart=0, entrystop = entrystop)
    df_other_original = df_other
    mask = df_other['nCustomAK8Puppi']>0
    df_other = df_other[mask]
    df_gen = tree.pandas.df(branches=gen_branches, entrystart=0, entrystop = entrystop)
    df_jet = tree.pandas.df(branches=jet_branches, entrystart=0, entrystop = entrystop)


    df_other.index = df_other.index+currententry
    df_gen.index = df_gen.index.set_levels(df_gen.index.levels[0]+currententry, level=0)
    df_jet.index = df_jet.index.set_levels(df_jet.index.levels[0]+currententry, level=0)
    currententry += len(df_other_original)

    df_others.append(df_other)
    df_gens.append(df_gen)
    df_jets.append(df_jet)
    
df_other = pd.concat(df_others)
df_gen = pd.concat(df_gens)
df_jet = pd.concat(df_jets)

# shuffle
df_other = df_other.sample(frac=1)
# apply new ordering to other dataframes
df_gen = df_gen.reindex(df_other.index.values,level=0)
df_jet = df_jet.reindex(df_other.index.values,level=0)

df_all  = df_jet.merge(df_gen,on='entry',how='outer')

dr = 0.8
deta = np.abs(df_all['CustomAK8Puppi_eta'] - df_all['LHEPart_eta'])
dphi = np.mod(df_all['CustomAK8Puppi_phi'] - df_all['LHEPart_phi'] + np.pi, 2*np.pi) - np.pi
df_all['dR'] = np.sqrt(deta*deta+dphi*dphi)
df_all = df_all.sort_values(['entry','CustomAK8Puppi_pt','dR'],ascending=[True,False,True])
df_all = df_all[df_all['dR'] < dr].drop_duplicates(subset=['CustomAK8Puppi_pt','CustomAK8Puppi_eta','CustomAK8Puppi_phi'])
df_jet = df_jet.reset_index().merge(df_all,how='left').set_index(['entry','subentry'])
df_jet['isHiggs'] = (np.abs(df_jet['LHEPart_pdgId'])==25).astype(int)

with tables.open_file(outfile, mode='w') as h5file:
    #max_gen = len(df_gen.index.get_level_values(-1).unique())
    max_jet = len(df_jet.index.get_level_values(-1).unique())

    #v_gen = _transform(df_gen, max_particles = max_gen)
    #for k in df_gen.columns:
    #    v = np.stack([v_gen[(k, i)].values for i in range(max_gen)], axis=-1)
    #    _write_carray(v, h5file, name=k)

    v_jet = _transform(df_jet, max_particles = max_jet)
    for k in df_jet.columns:
        v = np.stack([v_jet[(k, i)].values for i in range(max_jet)], axis=-1)
        _write_carray(v, h5file, name=k)
        
    for k in df_other.columns:
        _write_carray(df_other[k].values, h5file, name=k.replace('[','').replace(']',''))

f = tables.open_file(outfile)
print(f)
f.close()
