# Low-level numbering for multigrid support
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc
import dmplex

np.import_array()

include "dmplex.pxi"


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_cells(mc, mf):
    """Return a map from (renumbered) cells in a coarse mesh to those
    in a refined fine mesh.

    :arg mc: the coarse mesh to create the map from.
    :arg mf: the fine mesh to map to.
    :arg parents: a Section mapping original fine cell numbers to
         their corresponding coarse parent cells"""
    cdef:
        PETSc.DM cdm, fdm
        PetscInt fStart, fEnd, c, val, dim, nref, ncoarse
        PetscInt i, ccell, fcell, nfine
        np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine
        np.ndarray[PetscInt, ndim=1, mode="c"] co2n, cn2o, fo2n, fn2o

    cdm = mc._plex
    fdm = mf._plex
    dim = cdm.getDimension()
    nref = 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    co2n, cn2o  = dmplex.get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    fo2n, fn2o  = dmplex.get_entity_renumbering(fdm, mf._cell_numbering, "cell")
    coarse_to_fine = np.empty((ncoarse, nref), dtype=PETSc.IntType)
    coarse_to_fine[:] = -1

    # Walk owned fine cells:
    fStart, fEnd = 0, nfine
    for c in range(fStart, fEnd):
        # get original (overlapped) cell number
        fcell = fn2o[c]
        # The owned cells should map into non-overlapped cell numbers
        # (due to parallel growth strategy)
        assert fcell < fEnd

        # Find original coarse cell (fcell / nref) and then map
        # forward to renumbered coarse cell (again non-overlapped
        # cells should map into owned coarse cells)
        ccell = co2n[fcell / nref]
        assert ccell < ncoarse
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = c
                break
    return coarse_to_fine


@cython.boundscheck(False)
@cython.wraparound(False)
def p1_coarse_fine_map(Vc, Vf, np.ndarray[PetscInt, ndim=2, mode="c"] c2f_cells):
    """Build a map from a coarse P1 function space Vc to the finer space Vf.
    The resulting map is isomorphic in numbering to a P2 map in the coarse space.

    :arg Vc: The coarse space
    :arg Vf: The fine space
    :arg c2f_cells: The map from coarse cells to fine cells
    """
    cdef:
        PetscInt c, vStart, vEnd, i, ncoarse_cell, ndof, coarse_arity
        PetscInt l, j, k, tmp, other_cell, vfStart
        PetscInt coarse_vertex, fine_vertex, orig_coarse_vertex, ncell
        PetscInt coarse_shift, fine_shift
        PETSc.DM dm
        PETSc.PetscIS fpointIS
        bint done
        np.ndarray[np.int32_t, ndim=2, mode="c"] coarse_map, map_vals, fine_map
        np.ndarray[PetscInt, ndim=1, mode="c"] coarse_inv, fine_forward, orig_c2f

    coarse_mesh = Vc.mesh()
    fine_mesh = Vf.mesh()

    try:
        ndof, ncell = {'interval': (3, 2),
                       'triangle': (6, 4),
                       'tetrahedron': (10, 8)}[coarse_mesh.ufl_cell().cellname()]
    except KeyError:
        raise RuntimeError("Don't know how to make map")

    ncoarse_cell = Vc.mesh().cell_set.size
    map_vals = np.empty((ncoarse_cell, ndof), dtype=np.int32)
    map_vals[:, :] = -1
    coarse_map = Vc.cell_node_map().values
    fine_map = Vf.cell_node_map().values

    if not hasattr(coarse_mesh, '_vertex_new_to_old'):
        o, n = dmplex.get_entity_renumbering(coarse_mesh._plex, coarse_mesh._vertex_numbering, "vertex")
        coarse_mesh._vertex_old_to_new = o
        coarse_mesh._vertex_new_to_old = n

    coarse_inv = coarse_mesh._vertex_new_to_old

    if not hasattr(fine_mesh, '_vertex_new_to_old'):
        o, n = dmplex.get_entity_renumbering(fine_mesh._plex, fine_mesh._vertex_numbering, "vertex")
        fine_mesh._vertex_old_to_new = o
        fine_mesh._vertex_new_to_old = n

    fine_forward = fine_mesh._vertex_old_to_new

    vStart, vEnd = coarse_mesh._plex.getDepthStratum(0)
    vfStart, _ = fine_mesh._plex.getDepthStratum(0)
    # vertex numbers in the fine plex of coarse vertices
    orig_c2f = coarse_mesh._fpointIS.indices

    coarse_arity = coarse_map.shape[1]

    # Shift in plex points from non-overlapped to overlapped for
    # vertices (basically the additional cells)
    coarse_shift = coarse_mesh.cell_set.total_size - coarse_mesh.cell_set.size
    fine_shift = fine_mesh.cell_set.total_size - fine_mesh.cell_set.size
    for c in range(ncoarse_cell):
        # Find the fine vertices that correspond to coarse vertices
        # and put them sequentially in the first ndof map entries
        for i in range(coarse_arity):
            coarse_vertex = coarse_map[c, i]
            # Vertex numbers were shifted by the number of cells we
            # grew the halo by
            orig_coarse_vertex = coarse_inv[coarse_vertex] + vStart - coarse_shift
            fine_vertex = fine_forward[orig_c2f[orig_coarse_vertex] - vfStart + fine_shift]
            map_vals[c, i] = fine_vertex

        if coarse_arity == 2:
            # intervals, only one missing fine vertex dof
            # x----o----x
            done = False
            for j in range(ncell):
                for k in range(coarse_arity):
                    fine_vertex = fine_map[c2f_cells[c, j], k]
                    if fine_vertex != map_vals[c, 0] and fine_vertex != map_vals[c, 1]:
                        map_vals[c, 2] = fine_vertex
                        done = True
                        break
                if done:
                    break
        elif coarse_arity == 3:
            # triangles, 3 missing fine vertex dofs
            #
            #          x
            #         / \
            #        /   \
            #       o-----o
            #      / \   / \
            #     /   \ /   \
            #    x-----o-----x
            #
            for j in range(ncell):
                for k in range(coarse_arity):
                    fine_vertex = fine_map[c2f_cells[c, j], k]
                    done = False
                    # Original vertex or already found?
                    for i in range(coarse_arity):
                        if fine_vertex == map_vals[c, i] or \
                           fine_vertex == map_vals[c, i + 3]:
                            done = True
                            break
                    if done:
                        continue

                    other_cell = -1
                    # Find the cell this fine vertex is /not/ part of
                    for i in range(ncell):
                        done = False
                        for l in range(coarse_arity):
                            tmp = fine_map[c2f_cells[c, i], l]
                            if tmp == fine_vertex:
                                done = True
                        # Done is true if the fine_vertex was in the cell
                        if not done:
                            other_cell = i
                            break

                    # Now find the coarse vertex of of the cell we
                    # weren't in and put this fine vertex in the
                    # "opposite" position in the map
                    done = False
                    for l in range(coarse_arity):
                        tmp = fine_map[c2f_cells[c, other_cell], l]
                        for i in range(coarse_arity):
                            if tmp == map_vals[c, i]:
                                done = True
                                map_vals[c, i + 3] = fine_vertex
                                break
                        if done:
                            break

    return map_vals
