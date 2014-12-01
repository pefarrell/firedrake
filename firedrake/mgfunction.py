from pyop2 import op2
import pyop2.coffee.ast_base as ast

import function
import mgutils


__all__ = ["FunctionHierarchy"]


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
        self._dg0 = ((family == "OuterProductElement" and
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
