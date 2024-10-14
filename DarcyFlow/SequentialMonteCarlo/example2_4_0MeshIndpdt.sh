#!/bin/bash
echo "-------- Begin -----------"
mpiexec -n 20 python example2_4_2MeshDpdt.py --nx 20
mpiexec -n 20 python example2_4_2MeshDpdt.py --nx 40
mpiexec -n 20 python example2_4_2MeshDpdt.py --nx 60
mpiexec -n 20 python example2_4_2MeshDpdt.py --nx 80
mpiexec -n 20 python example2_4_2MeshDpdt.py --nx 100

mpiexec -n 20 python example2_4_1MeshIndpdt.py --nx 20
mpiexec -n 20 python example2_4_1MeshIndpdt.py --nx 40
mpiexec -n 20 python example2_4_1MeshIndpdt.py --nx 60
mpiexec -n 20 python example2_4_1MeshIndpdt.py --nx 80
mpiexec -n 20 python example2_4_1MeshIndpdt.py --nx 100

python example2_4_3PlotMeshIndpdt.py