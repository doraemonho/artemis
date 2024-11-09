# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Regression to test a ssheet in a 2D disk

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
import scripts.ssheet.ssheet as ssheet

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def get_mpi_slots():
    try:
        result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True, check=True)
        sockets = 0
        for line in result.stdout.split('\n'):
            if 'Socket(s):' in line:
                sockets = int(line.split(':')[1].strip())
        return sockets * 2  # Assuming 2 ranks per socket
    except Exception as e:
        print(f"Error getting MPI slot info: {e}")
        return 2  # Default to 2 slots if detection fails

#ssheet._nranks = min(max(2, os.cpu_count()), 8)
ssheet._nranks = min(max(2, get_mpi_slots()), 8)
print(f'slots: {get_mpi_slots()}')
print(f'os cpu count: {os.cpu_count()}')
print(f'ranks: {ssheet._nranks}')
ssheet._file_id = "ssheet_mpi"


# Run Artemis
def run(**kwargs):
    return ssheet.run(**kwargs)


# Analyze outputs
def analyze():
    return ssheet.analyze()
