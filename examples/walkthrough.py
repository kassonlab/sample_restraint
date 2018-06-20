#!/usr/bin/env python

import sys
import os
import gmx
import numpy
import subprocess
import matplotlib
from matplotlib import pyplot as plt

# This only works if the gmx binary path was set in the parent process before launching the Jupyter server.
def find_program(program):
    """Return the first occurrence of program in PATH or None if not found."""
    for path in os.environ["PATH"].split(os.pathsep):
        fpath = os.path.join(path, program)
        if os.path.isfile(fpath) and os.access(fpath, os.X_OK):
            return fpath
    return None
gmx_path = find_program("gmx")
if gmx_path is None:
    gmx_path = find_program("gmx_mpi")
if gmx_path is None:
    raise UserWarning("gmx executable not found in path.")

datadir = os.path.abspath('alanine-dipeptide')
workingdir = os.path.basename(datadir)

for structure in range(4):
    structure_file = os.path.join(datadir, 'equil{}.gro'.format(structure))
    tpr_file = os.path.join(datadir, 'input{}.tpr'.format(structure))
    grompp_args = ['-c', structure_file,
                   '-o', tpr_file,
                   '-f', os.path.join(datadir, 'grompp.mdp'),
                   '-p', os.path.join(datadir, 'topol.top')]
    subprocess.call([gmx_path, "grompp"] + grompp_args)

tpr_files = [os.path.join(datadir, 'input{}.tpr'.format(i)) for i in range(4)]
md = gmx.workflow.from_tpr(input=tpr_files, grid=[1,1,1])

my_context = gmx.context.ParallelArrayContext(md)

with my_context as session:
   session.run()

# Wrap the gmx tool to extract phi and psi values for a Ramachandran diagram. E.g.
# ~/gromacs-mpi/bin/gmx_mpi rama -s topol.tpr -f traj_comp.xtc
def rama(run_input=None, trajectory=None, output="rama.xvg", executable=gmx_path):
    """Use the GROMACS tool to extract psi and phi angles for the provided structure and trajectory."""
    if run_input is not None and trajectory is not None and output is not None and executable is not None:
        for file_arg in [run_input, trajectory]:
            if not os.path.exists(file_arg):
                raise RuntimeError("Invalid file: {}".format(file_arg))
        if not os.access(gmx_path, os.X_OK):
            raise RuntimeError("Invalid executable: {}".format(gmx_path))
    else:
        raise RuntimeError("Bad arguments.")
    subprocess.call([gmx_path, "rama", "-s", run_input, "-f", trajectory, "-o", output])


run_input = os.path.join(datadir, 'input{}.tpr'.format(my_context.rank))
trajectory = os.path.join(my_context.workdir, 'traj_comp.xtc')
trajectory = os.path.join(my_context.workdir, 'traj.trr')

rama_file = os.path.join(my_context.workdir, 'rama.xvg')
rama(run_input=run_input, trajectory=trajectory, output=rama_file)

phi, psi = numpy.genfromtxt(rama_file, skip_header=13, comments='@', usecols=(0,1)).T
