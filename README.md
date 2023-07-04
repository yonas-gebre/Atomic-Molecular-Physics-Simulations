# Time-dependent Schrödinger equation solver for atomic and diatomic systems
This project focuses on modeling the interaction between atoms and molecules with intense laser pulses. Its primary goal is to solve the time-dependent Schrödinger equation (TDSE) for atoms and diatomic molecules exposed to high-intensity, ultrafast laser pulses. By solving the TDSE, valuable insights can be gained into processes such as electron excitation and deexcitation through photon absorption, electron ionization from atoms and molecules, and high harmonic generation (HHG).






1. [ Dependencies. ](#desc)
2. [ Usage. ](#usage)
3. [ Credits. ](#development)


<a name="desc"></a>
## 1. Dependencies

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
