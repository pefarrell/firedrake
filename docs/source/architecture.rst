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

.. _architecture-data-structures:

Data Structures
---------------

.. _architecture-meshes:

Meshes
~~~~~~

A :class:`~firedrake.core_types.Mesh` is the representation of the abstract
topology and concrete geometry of an unstructured mesh. The topology is
described by :class:`Sets <pyop2.Set>` and :class:`Maps <pyop2.Map>` between
them.  Geometric information, that is the coordinates of vertices in 2D or 3D
space, is stored as an ordinary :class:`~firedrake.types.Function`. Not
treating the coordinates special allows reasoning about and manipulating the
coordinate field like any other field as described `below
<architecture-functions_>`_.

.. TODO: Plex

.. _architecture-function-spaces:

Function Spaces
~~~~~~~~~~~~~~~

A :class:`~firedrake.types.FunctionSpace` represents a chosen discretisation
described by a :class:`~ufl.finiteelement.finiteelement.FiniteElement` on a
given :class:`~firedrake.core_types.Mesh` and thereby defines a
:class:`~pyop2.Set` of degrees of freedom and a :class:`~pyop2.Map` to the
:class:`~pyop2.Set` of cells of the mesh. Firedrake distinguishes a
:class:`~firedrake.types.VectorFunctionSpace` defined on a
:class:`~ufl.finiteelement.mixedelement.VectorElement` and a
:class:`~firedrake.types.MixedFunctionSpace` on a
:class:`~ufl.finiteelement.mixedelement.MixedElement` or more conveniently
built by combining two function spaces.

.. TODO: IndexedFunctionSpace

.. _architecture-functions:

Functions
~~~~~~~~~

A :class:`~firedrake.types.Function` represents a field defined on a given
:class:`~firedrake.types.FunctionSpace` and carries its data in form of a
:class:`~pyop2.Dat`. That means the data is stored in an abstract vector
managed entirely by PyOP2_ and might live on a GPU depending on the chosen
PyOP2_ backend. Firedrake and the user can therefore reason about the data in
an abstract way without concern for data layout or transfer and all of
PyOP2_'s functionality is at their disposal for manipulating functions.
Functions support common operations on fields such as addition, subtraction,
and multiplication or division by a scalar via overloaded operators.

At the same time a :class:`~firedrake.types.Function` represents a
:class:`~ufl.coefficient.Coefficient` and can be used as such in a UFL_ form.

.. _architecture-expressions:

Expressions
~~~~~~~~~~~

An :class:`~firedrake.expression.Expression` is a code snippet evaluated as the
right-hand side expression on a :class:`~firedrake.types.FunctionSpace`. It can
contain mathematical functions and has access to the coordinate field.

Expressions can be :meth:`interpolated <firedrake.types.Function.interpolate>`
onto Functions which are defined on a :class:`~firedrake.types.FunctionSpace`
that supports point evaluation or :func:`projected
<firedrake.projection.project>` onto any
:class:`~firedrake.types.FunctionSpace` to produce a
:class:`~firedrake.types.Function` on that space.

The code snippet defining an expression is used to produce a
:class:`~pyop2.Kernel` evaluating the expression on the :class:`~pyop2.Dat`
containing the data of the target :class:`~firedrake.types.Function` using a
:func:`~pyop2.par_loop`. The actual backend-specific computation is managed by
PyOP2_ and Firedrake merely needs to supply the kernel.

A heavily simplified version of the
:meth:`~firedrake.types.Function.interpolate` method is given below: ::

    def interpolate(self, expression, subset=None):
        fs = self.function_space()
        coords = fs.mesh().coordinates

        # ... generate kernel_code from expression

        kernel = op2.Kernel(kernel_code, "expression_kernel")

        op2.par_loop(kernel, subset or self.cell_set,
                     self.dat(op2.WRITE, fs.cell_node_map()[op2.i[0]]),
                     coords.dat(op2.READ, coords.cell_node_map())
                     )

.. _evtk: https://bitbucket.org/pauloh/pyevtk
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _PyOP2: http://op2.github.io/PyOP2
.. _UFL: https://bitbucket.org/mapdes/ufl
.. _VTK: http://vtk.org
