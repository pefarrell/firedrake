import numpy as np


from pyop2 import op2
import pyop2.coffee.ast_base as ast

import dmplex
import mgimpl
import mgutils
import function
import functionspace
import mesh


__all__ = ['MeshHierarchy', 'FunctionSpaceHierarchy', 'FunctionHierarchy',
           'ExtrudedMeshHierarchy']


class MeshHierarchy(mesh.Mesh):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh"""
    def __init__(self, m, refinement_levels, reorder=None):
        """
        :arg m: the coarse :class:`~.Mesh` to refine
        :arg refinement_levels: the number of levels of refinement
        :arg reorder: optional flag indicating whether to reorder the
             refined meshes.
        """
        m._plex.setRefinementUniform(True)
        dm_hierarchy = []

        dm = m._plex
        fpoint_ises = []
        for i in range(refinement_levels):
            rdm = dm.refine()
            fpoint_ises.append(dm.createCoarsePointIS())
            # Remove interior facet label (re-construct from
            # complement of exterior facets).  Necessary because the
            # refinement just marks points "underneath" the refined
            # facet with the appropriate label.  This works for
            # exterior, but not marked interior facets
            rdm.removeLabel("interior_facets")
            # Remove vertex (and edge) points from labels on exterior
            # facets.  Interior facets will be relabeled in Mesh
            # construction below.
            dmplex.filter_exterior_facet_labels(rdm)
            rdm.removeLabel("op2_core")
            rdm.removeLabel("op2_non_core")
            rdm.removeLabel("op2_exec_halo")
            rdm.removeLabel("op2_non_exec_halo")

            dm_hierarchy.append(rdm)
            dm = rdm
            # Fix up coords if refining embedded circle or sphere
            if hasattr(m, '_circle_manifold'):
                coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
                scale = m._circle_manifold / np.linalg.norm(coords, axis=1).reshape(-1, 1)
                coords *= scale
            elif hasattr(m, '_icosahedral_sphere'):
                coords = dm.getCoordinatesLocal().array.reshape(-1, 3)
                scale = m._icosahedral_sphere / np.linalg.norm(coords, axis=1).reshape(-1, 1)
                coords *= scale

        m._init()
        self._hierarchy = [m] + [mesh.Mesh(dm, dim=m.ufl_cell().geometric_dimension(),
                                           distribute=False, reorder=reorder)
                                 for i, dm in enumerate(dm_hierarchy)]

        self._ufl_cell = m.ufl_cell()
        self._cells_vperm = []
        for m in self:
            m._init()

        for mc, mf, fpointis in zip(self._hierarchy[:-1],
                                    self._hierarchy[1:],
                                    fpoint_ises):
            mc._fpointIS = fpointis
            c2f = mgimpl.coarse_to_fine_cells(mc, mf)
            P1c = functionspace.FunctionSpace(mc, 'CG', 1)
            P1f = functionspace.FunctionSpace(mf, 'CG', 1)
            self._cells_vperm.append(mgimpl.orient_cells(P1c, P1f, c2f))

    def __iter__(self):
        for m in self._hierarchy:
            yield m

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]


class ExtrudedMeshHierarchy(MeshHierarchy):
    def __init__(self, mesh_hierarchy, layers, kernel=None, layer_height=None,
                 extrusion_type='uniform', gdim=None):
        self._base_hierarchy = mesh_hierarchy
        self._hierarchy = [mesh.ExtrudedMesh(m, layers, kernel=kernel,
                                             layer_height=layer_height,
                                             extrusion_type=extrusion_type,
                                             gdim=gdim)
                           for m in mesh_hierarchy]
        self._ufl_cell = self[0].ufl_cell()
        self._cells_vperm = self._hierarchy._cells_vperm


class FunctionSpaceHierarchy(object):
    """Build a hierarchy of function spaces.

    Given a hierarchy of meshes, this constructs a hierarchy of
    function spaces, with the property that every coarse space is a
    subspace of the fine spaces that are a refinement of it.
    """
    def __init__(self, mesh_hierarchy, family, degree=None,
                 name=None, vfamily=None, vdegree=None):
        """
        :arg mesh_hierarchy: a :class:`.MeshHierarchy` to build the
             function spaces on.
        :arg family: the function space family
        :arg degree: the degree of the function space
        """
        self._mesh_hierarchy = mesh_hierarchy
        self._hierarchy = [functionspace.FunctionSpace(m, family, degree=degree,
                                                       name=name, vfamily=vfamily,
                                                       vdegree=vdegree)
                           for m in self._mesh_hierarchy]

        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()
        self._restriction_weights = None

    def __len__(self):
        return len(self._hierarchy)

    def __iter__(self):
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        return self._hierarchy[idx]

    def ufl_element(self):
        return self._ufl_element

    def cell_node_map(self, level, bcs=None):
        """A :class:`pyop2.Map` from cells on a coarse mesh to the
        corresponding degrees of freedom on a the fine mesh below it.

        :arg level: the coarse level the map should be from.
        :arg bcs: optional iterable of :class:`.DirichletBC`\s
             (currently ignored).
        """
        if not 0 <= level < len(self) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self) - 1))
        try:
            return self._map_cache[level]
        except KeyError:
            pass
        Vc = self._hierarchy[level]
        Vf = self._hierarchy[level + 1]

        element = self.ufl_element()
        family = element.family()
        degree = element.degree()

        c2f, vperm = self._mesh_hierarchy._cells_vperm[level]

        if isinstance(self._mesh_hierarchy, ExtrudedMeshHierarchy):
            if not (element._A.family() == "Discontinuous Lagrange" and
                    element._B.family() == "Discontinuous Lagrange" and
                    degree == (0, 0)):
                raise NotImplementedError
            arity = Vf.cell_node_map().arity * c2f.shape[1]
            map_vals = Vf.cell_node_map().values[c2f].flatten()
            offset = np.repeat(Vf.cell_node_map().offset, c2f.shape[1])
            map = op2.Map(self._cell_sets[level],
                          Vf.node_set,
                          arity,
                          map_vals,
                          offset=offset)
            self._map_cache[level] = map
            return map

        map_vals = mgimpl.create_cell_node_map(Vc, Vf, c2f, vperm)
        map = op2.Map(self._cell_sets[level],
                      Vf.node_set,
                      map_vals.shape[1],
                      map_vals)
        self._map_cache[level] = map
        return map


class FunctionHierarchy(object):
    """Build a hierarchy of :class:`~.Function`\s"""
    def __init__(self, fs_hierarchy):
        """
        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.

        `fs_hierarchy` may also be an existing
        :class:`FunctionHierarchy`, in which case a copy of the
        hierarchy is returned.
        """
        if isinstance(fs_hierarchy, FunctionHierarchy):
            self._function_space = fs_hierarchy.function_space()
        else:
            self._function_space = fs_hierarchy

        self._hierarchy = [function.Function(f) for f in fs_hierarchy]

        element = self._function_space[0].ufl_element()
        family = element.family()
        degree = element.degree()
        self._dg0 = ((family == "OuterProductElement" and \
                      (element._A.family() == "Discontinuous Lagrange" and
                       element._B.family() == "Discontinuous Lagrange" and
                       degree == (0, 0))) or
                     (family == "Discontinuous Lagrange" and degree == 0))

        if not self._dg0:
            element = self._function_space[0].fiat_element
            omap = self[1].cell_node_map().values
            c2f, vperm = self._function_space._mesh_hierarchy._cells_vperm[0]
            indices = mgutils.get_unique_indices(element,
                                                 omap[c2f[0, :], ...].reshape(-1),
                                                 vperm[0, :])
            self._prolong_kernel = mgutils.get_prolongation_kernel(element, indices)
            self._restrict_kernel = mgutils.get_restriction_kernel(element, indices)
            self._inject_kernel = mgutils.get_injection_kernel(element, indices)

    def __iter__(self):
        for f in self._hierarchy:
            yield f

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]

    def function_space(self):
        return self._function_space

    def cell_node_map(self, i):
        return self._function_space.cell_node_map(i)

    def prolong(self, level):
        """Prolong from a coarse to the next finest hierarchy level.

        :arg level: The coarse level to prolong from"""

        if not 0 <= level < len(self) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self) - 1))
        if self._dg0:
            self._prolong_dg0(level)
            return
        coarse = self[level]
        fine = self[level+1]
        op2.par_loop(self._prolong_kernel, self.function_space()._cell_sets[level],
                     fine.dat(op2.WRITE, self.cell_node_map(level)),
                     coarse.dat(op2.READ, coarse.cell_node_map()))

    def restrict(self, level, is_solution=False):
        """Restrict from a fine to the next coarsest hierarchy level.

        :arg level: The fine level to restrict from
        :kwarg is_solution: optional keyword argument indicating if
            the :class:`~.Function` being restricted is a *solution*,
            living in the primal space or a *residual* (cofunction)
            living in the dual space (the default).  Residual
            restriction is weighted by the size of the coarse cell
            relative to the fine cells (i.e. the mass matrix) whereas
            solution restriction need not be weighted."""

        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        if self._dg0:
            self._restrict_dg0(level, is_solution=is_solution)
            return
        fs = self.function_space()
        if fs._restriction_weights is None:
            fs._restriction_weights = FunctionHierarchy(fs)
            k = """
            static inline void weights(double weight[%d])
            {
                for ( int i = 0; i < %d; i++ ) {
                    weight[i] += 1.0;
                }
            }""" % (self.cell_node_map(0).arity, self.cell_node_map(0).arity)
            fn = fs._restriction_weights
            k = op2.Kernel(k, 'weights')
            # Count number of times cell loop hits
            for lvl in range(1, len(fn)):
                op2.par_loop(k, self.function_space()._cell_sets[lvl-1],
                             fn[lvl].dat(op2.INC, fn.cell_node_map(lvl-1)[op2.i[0]]))
                # Inverse, since we're using these as weights, not
                # counts.
                fn[lvl].assign(1.0/fn[lvl])
        coarse = self[level-1]
        fine = self[level]
        weights = fs._restriction_weights[level]
        coarse.dat.zero()
        op2.par_loop(self._restrict_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]], flatten=True),
                     fine.dat(op2.READ, self.cell_node_map(level-1), flatten=True),
                     weights.dat(op2.READ, self.cell_node_map(level-1), flatten=True))

    def inject(self, level):
        """Inject from a fine to the next coarsest hierarchy level.

        :arg level: the fine level to inject from"""
        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        coarse = self[level-1]
        fine = self[level]
        op2.par_loop(self._inject_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.WRITE, coarse.cell_node_map()),
                     fine.dat(op2.READ, self.cell_node_map(level-1)))

    def _prolong_dg0(self, level):
        c2f_map = self.cell_node_map(level)
        coarse = self[level]
        fine = self[level + 1]
        if not hasattr(self, '_prolong_kernel'):
            k = ast.FunDecl("void", "prolong_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.c_for("fdof", c2f_map.arity,
                                           ast.Assign(ast.Symbol("fine", ("fdof", 0)),
                                                      ast.Symbol("coarse", (0, 0))),
                                           pragma=None),
                            pred=["static", "inline"])
            self._prolong_kernel = op2.Kernel(k, "prolong_dg0")
        op2.par_loop(self._prolong_kernel, self.function_space()._cell_sets[level],
                     coarse.dat(op2.READ, coarse.cell_node_map()),
                     fine.dat(op2.WRITE, c2f_map))

    def _restrict_dg0(self, level, is_solution=False):
        c2f_map = self.cell_node_map(level - 1)
        coarse = self[level - 1]
        fine = self[level]
        if not hasattr(self, '_restrict_kernel'):
            if is_solution:
                detJ = 1.0/c2f_map.arity
            else:
                detJ = 1.0
            k = ast.FunDecl("void", "restrict_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.Block([ast.Decl(coarse.dat.ctype, "tmp", init=0.0),
                                            ast.c_for("fdof", c2f_map.arity,
                                                      ast.Incr(ast.Symbol("tmp"),
                                                               ast.Symbol("fine", ("fdof", 0))),
                                                      pragma=None),
                                            ast.Assign(ast.Symbol("coarse", (0, 0)),
                                                       ast.Prod(detJ, ast.Symbol("tmp")))]),
                            pred=["static", "inline"])
            self._restrict_kernel = op2.Kernel(k, "restrict_dg0")

        op2.par_loop(self._restrict_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.WRITE, coarse.cell_node_map()),
                     fine.dat(op2.READ, c2f_map))
