.. contents::

.. _architecture:

Architecture
============

Firedrake composes a variety of building blocks to form a framework for
solving partial differential equations with the finite element method. These
components and their responsibilities are listed below.

* The Unified Form Language (UFL_) is used to describe variational forms and
  their discretisations.
* The FEniCS form compiler (FFC_) translates variational forms into numerical
  kernels describing the local assembly operations.
* The FInite element Automatic Tabulator (FIAT_) is called by FFC_ to tabulate
  finite element basis functions and their derivatives.
* PETSc_ provides linear and nonlinear solvers, preconditioners, matrix and
  vector data structures and a parallel mesh builder.
* evtk_ is used for writing out fields to files in the popular VTK_ format.
* PyOP2_ executes finite element assembly kernels in parallel on a variety of
  hardware platforms, describes the mesh topology and abstracts the data
  storage for fields and matrices.

.. _evtk: https://bitbucket.org/pauloh/pyevtk
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _PyOP2: http://op2.github.io/PyOP2
.. _UFL: https://bitbucket.org/mapdes/ufl
.. _VTK: http://vtk.org
