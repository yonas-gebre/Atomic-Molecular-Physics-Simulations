# Time-dependent Schrödinger equation solver for atomic and diatomic systems
This project centers around developing software designed to perform simulations that delve into the interactions between atoms and molecules with laser pulses within a quantum mechanical framework. The primary objective is to solve the time-dependent Schrödinger equation (TDSE) for atoms and diatomic molecules under the influence of high-intensity, ultrafast laser pulses.

1. [ Dependencies. ](#desc)
2. [ Usage. ](#usage)
3. [ Credits. ](#development)


<a name="desc"></a>
## 1. Description

PETSC, SLEPC, petsc4py, and slepc4py with complex number

mpi4py

<a name="usage"></a>
## 2. Usage

The code can be run by using the TISE.py file inside the directory source_code. To run with 12 processors one writes:

mpiexec -n 12 TISE.py

The file requires the "input.json" file with the parameters for the calculations to be present. Please see doc/developer_note.pdf for the detailed explanation of the parameters in the "input.json" file.


<a name="development"></a>
## 3. Credits
This project was done with work funded by the  National Science Foundation (NSF) and the U.S. Department of Energy. It was conducted under the guidance of Andreas Becker, a research professor at JILA, University of Colorado Boulder. Joel Venzke, a former graduate student, made significant contributions to this endeavor.
