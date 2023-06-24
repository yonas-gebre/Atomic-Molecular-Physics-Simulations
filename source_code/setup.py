from distutils.core import setup
from Cython.Build import cythonize
##CC=gcc python setup.py build_ext --inplace
setup(
    ext_modules = cythonize(["Interaction_Matrix.pyx","Propagator.pyx","DC_Potential.pyx", "Interaction_Matrix.pyx", "Dipole_Acceleration_Matrix.pyx", "SC_Field_Free_Hamiltonian.pyx", "DC_Field_Free_Hamiltonian.pyx", "Propagate_Linear.pyx"])
)