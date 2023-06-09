import mdtraj as md
import pandas as pd
import os
import numpy as np
import glob
from itertools import combinations
from joblib import Parallel,delayed
import scipy.optimize as opt

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

### define some functions,Q for Tm,hydrophobicSASA for Tagg
def best_hummer_q(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 3])
    
    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    print("Number of native contacts", len(native_contacts))
    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q 

### define some functions,Q for Tm,hydrophobicSASA for Tagg
def best_hummer_q_hydrophilic_All(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    hydrophilic_atoms = native.topology.select('mass 3 to 50 and code R N D Q E K H')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - native.topology.atom(j).residue.index) > 3 \
                and i in hydrophilic_atoms])
    
    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    print("Number of native contacts", len(native_contacts))
    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q 
def get_hydrophobic_SASA(t):
    # pdb = wdir+case+'/5_run/archive/mol.pdb'
    # dcd = wdir+case+'/5_run/archive/movie.dcd'
    # t = md.load(dcd,top=pdb)
    # t=t[::10][0:950] #等间隔10frame取一帧，以满足10frame/ns
    # #数据密度不足，可能导致拟合失败。
    top = t.topology
    res = [str(i)[0:3] for i in list(top.residues)]
    print(res)

    res_map = []
    for i in res:
        if i in ['TRP','TYR','PHE','MET','LEU','ILE','VAL','ALA']:
            res_map.append('A')
        else:
            res_map.append('B')

    sasa = md.shrake_rupley(t,mode='residue')

    df = pd.DataFrame(sasa)

    df.columns = res_map
    
    df['T'] = [i*0.3+300 for i in range(len(t))]
    df.set_index('T')

    df_ = df.T
    df_ = df_[df_.index.isin(['A'])]

    df_o = pd.DataFrame(df_.mean(), columns=['hydrophobic SASA'])
    # df_o.to_csv(case+'_hydrophobic_SASA.csv')
    return list(df_o['hydrophobic SASA'])

### curve fit sigmoid function
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(c * (x - d))) + b
def sigmoid_fit(x,y):
    (a_, b_, c_, d_), _ = opt.curve_fit(sigmoid, x, y,[30,0,0.5,0.7],maxfev=100000)
    y_fit = sigmoid(x, a_, b_, c_, d_)
    # print(len(y_fit))
    return y_fit,d_
### curve fit exp function
def f_exp(x, a, b, c, d):
    return a * np.exp(c * (x - d)) + b
def f_exp_fit(x,y):
    (a_, b_, c_, d_), _ = opt.curve_fit(f_exp, x, y,[30,0,0.5,0.7],maxfev=100000)
    y_fit = f_exp(x, a_, b_, c_, d_)
    # print(len(y_fit))
    return y_fit,d_

### analysis all 

def case2traj(case):
    pdb = os.path.join(case,'5_run/archive/mol.pdb')
    dcd = os.path.join(case,'5_run/archive/movie.dcd')
    t = md.load(dcd,top=pdb)
    t=t[::10] #等间隔10frame取一帧，以满足10frame/ns。数据量恰当，才能fit出合适的曲线。
    return t
def analysis(case):
    # input t is the mdtraj trajectory object from function case2traj
    t = case2traj(case)

    df = pd.DataFrame()
    # df['case'] = case
    df['XCelDev_T'] = [i/3000 for i in range(len(t))]
    df['RMSD'] = md.rmsd(t,t,0) 
    df['Rg'] = md.compute_rg(t)

    df['hydrophobicSASA'] = get_hydrophobic_SASA(t)
    df['hydrophobicSASA_fit'] = f_exp_fit(df['XCelDev_T'],df['hydrophobicSASA'])[0]
    Tagg = f_exp_fit(df['XCelDev_T'],df['hydrophobicSASA'])[1]

    df['Q(fraction of native contact)'] = best_hummer_q(t, t[0])
    df['Q_fit'] = sigmoid_fit(df['XCelDev_T'],df['Q(fraction of native contact)'])[0]
    Tm = sigmoid_fit(df['XCelDev_T'],df['Q(fraction of native contact)'])[1]

    # df[0:296].to_csv(case+'.csv',index=False)
    return df,Tm,Tagg
def post_put(case):
   df,Tm,Tagg = analysis(case) 
   f_name = case.split('/')[-1]
   df.to_csv(f_name+'_data.csv',index=False)
   f = open(f_name+'_Tm_Tagg.dat','w')
   f.write('case,Tm,Tagg\n'+','.join([f_name,str(Tm),str(Tagg)]))
   f.close()


### plot
def plt_scatter_corr(a,b,t):
    # g = sns.jointplot(x=a, y=b, kind='reg',palette=c)
    g = sns.jointplot(x=a, y=b, kind='reg') 
    # ax.annotate(stats.pearsonr)
    r, p = stats.pearsonr(a, b)
    g.ax_joint.annotate(f'$pearson corr = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
    r1, p1 = stats.spearmanr(a, b)
    g.ax_joint.annotate(f'$spearman corr = {r1:.3f}$',
                    xy=(0.1, 0.8), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
    g.ax_joint.scatter(a, b)
    # g.set_axis_labels(xlabel='a', ylabel='b', size=15)
    plt.tight_layout()
    plt.title(t)
    plt.savefig(t+'.png',dpi=300)
    plt.show()

