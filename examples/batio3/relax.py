from chgnet.model import StructOptimizer
from ase.io import read,write
from datetime import  datetime as dt 
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from monty.serialization import loadfn,dumpfn
import sys 
from pymatgen.io.ase import AseAtomsAdaptor


pressure = float(sys.argv[1])
seed = sys.argv[2]
atoms = read(seed + '.xyz')

def benchmarkrelax(ase_structure=None,
          struct_optimiser_kwargs={'use_device':'cpu',
                                   'optimizer_class':'FIRE'
                                   },
          relax_kwargs={'steps':200,
                        'fmax':0.01,
                        'verbose':True,
                        }):
    relaxer = StructOptimizer(**struct_optimiser_kwargs)
    result = relaxer.relax(ase_structure,**relax_kwargs)
    fmax = np.max(
        [np.linalg.norm(x) for x in result['trajectory'].forces[-1]]
        )
    return(result)

result= benchmarkrelax(ase_structure = atoms)

final_structure = result['final_structure']

if result:
    aseatoms = AseAtomsAdaptor.get_atoms(final_structure)
    aseatoms.calc = None
    aseatoms.write(filename='{}-out.xyz'.format(seed),format='extxyz')
    volume = aseatoms.get_volume() 
    enthalpy = result['trajectory'].energies[-1] + pressure*volume
    with open(seed + '.castep','w') as f:
        f.write("Current cell volume = {:25.15f} A**3\n".format(volume))
        f.write("*  Pressure:   {:25.15f}\n".format(pressure))
        f.write("Python: Final Enthalpy     = {:25.15f} eV\n".format(enthalpy))
