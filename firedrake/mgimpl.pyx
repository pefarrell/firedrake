# Low-level numbering for multigrid support
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc
import dmplex
import mgutils

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


@cython.wraparound(False)
@cython.boundscheck(False)
def orient_cells(P1c, P1f, np.ndarray[PetscInt, ndim=2, mode="c"] c2f):
    cdef:
        PetscInt vcStart, vcEnd, vfStart, vfEnd, cshift, fshift,
        PetscInt ncoarse, nfine, ccell, fcell, i, j, k, vtx, ovtx, fcvtx, cvtx
        PetscInt nvertex, ofcell
        bint found
        np.ndarray[PetscInt, ndim=2, mode="c"] new_c2f = -np.ones_like(c2f)
        np.ndarray[PetscInt, ndim=1, mode="c"] inv_cvertex, fvertex, indices
        np.ndarray[PetscInt, ndim=2, mode="c"] cvertices, fvertices
        np.ndarray[PetscInt, ndim=2, mode="c"] vertex_perm
    coarse = P1c.mesh()
    fine = P1f.mesh()

    ncoarse = coarse.cell_set.size
    nfine = fine.cell_set.size
    cshift = coarse.cell_set.total_size - ncoarse
    fshift = fine.cell_set.total_size - nfine

    vcStart, vcEnd = coarse._plex.getDepthStratum(0)
    vfStart, vfEnd = fine._plex.getDepthStratum(0)

    # Get renumbering to original (plex) vertex numbers
    _, inv_cvertex = dmplex.get_entity_renumbering(coarse._plex,
                                                   coarse._vertex_numbering, "vertex")
    fvertex, _ = dmplex.get_entity_renumbering(fine._plex,
                                               fine._vertex_numbering, "vertex")

    # Get map from coarse points into corresponding fine mesh points.
    # Note this is only valid for "owned" entities (non-overlapped)
    indices = coarse._fpointIS.indices

    cvertices = P1c.cell_node_map().values

    fmap = P1f.cell_node_map().values
    fvertices = P1f.cell_node_map().values
    nvertex = P1c.cell_node_map().arity
    vertex_perm = -np.ones((ncoarse, nvertex*4), dtype=PETSc.IntType)
    for ccell in range(ncoarse):
        for fcell in range(4):
            # Cell order (given coarse reference cell numbering) is as below:
            # 2
            # |\
            # | \
            # |  \
            # | 2 \
            # *----*
            # |\ 3 |\
            # | \  | \
            # |  \ |  \
            # | 0 \| 1 \
            # 0----*----1
            #
            found = False
            # Check if this cell shares a vertex with the coarse cell
            # In which case, if it shares coarse vertex i it is in position i.
            for j in range(nvertex):
                if found:
                    break
                vtx = fvertices[c2f[ccell, fcell], j]
                for i in range(nvertex):
                    cvtx = cvertices[ccell, i]
                    fcvtx = fvertex[indices[inv_cvertex[cvtx] + vcStart - cshift]
                                    - vfStart + fshift]
                    if vtx == fcvtx:
                        new_c2f[ccell, i] = c2f[ccell, fcell]
                        found = True
                        break
            # Doesn't share any vertices, must be central cell, which comes last.
            if not found:
                new_c2f[ccell, 3] = c2f[ccell, fcell]
        # Having computed the fine cell ordering on this coarse cell,
        # we derive the permutation of each fine cell vertex.
        # Vertex order on each fine cell is given by:
        # 2
        # |\
        # | \
        # |  \
        # | 2 \
        # b----a
        # |\ 3 |\
        # | \  | \
        # |  \ |  \
        # | 0 \| 1 \
        # 0----c----1
        #
        # 0_f => [0, c, b]
        # 1_f => [1, a, c]
        # 2_f => [2, b, a]
        # 3_f => [a, b, c]
        #
        for fcell in range(3):
            # "Other" cell, vertex neither shared with coarse cell
            # vertex nor this cell is vertex 1, (the shared vertex is
            # vertex 2).
            ofcell = (fcell + 2) % 3
            for i in range(nvertex):
                vtx = fvertices[new_c2f[ccell, fcell], i]
                # Is this vertex shared with the coarse grid?
                found = False
                for j in range(nvertex):
                    cvtx = cvertices[ccell, j]
                    fcvtx = fvertex[indices[inv_cvertex[cvtx] + vcStart - cshift]
                                    - vfStart + fshift]
                    if vtx == fcvtx:
                        found = True
                        break
                if found:
                    # Yes, this is vertex 0.
                    vertex_perm[ccell, fcell*nvertex + i] = 0
                    continue

                # Is this vertex shared with "other" cell
                found = False
                for j in range(nvertex):
                    ovtx = fvertices[new_c2f[ccell, ofcell], j]
                    if vtx == ovtx:
                        found = True
                        break
                if found:
                    # Yes, this is vertex 2.
                    vertex_perm[ccell, fcell*nvertex + i] = 2
                    # Find vertex in cell 3 that matches this one.
                    # It is numbered by the other cell's other
                    # cell.
                    for j in range(nvertex):
                        ovtx = fvertices[new_c2f[ccell, 3], j]
                        if vtx == ovtx:
                            vertex_perm[ccell, 3*nvertex + j] = (fcell + 4) % 3
                            break
                if not found:
                    # No, this is vertex 1
                    vertex_perm[ccell, fcell*nvertex + i] = 1
                    # Find vertex in cell 3 that matches this one.
                    # It is numbered by the "other" cell
                    for j in range(nvertex):
                        ovtx = fvertices[new_c2f[ccell, 3], j]
                        if vtx == ovtx:
                            vertex_perm[ccell, 3*nvertex + j] = ofcell
                            break
    return new_c2f, vertex_perm


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline PetscInt hash_perm(PetscInt p0, PetscInt p1):
    if p0 == 0:
        if p1 == 1:
            return 0
        return 1
    if p0 == 1:
        if p1 == 0:
            return 2
        return 3
    if p0 == 2:
        if p1 == 0:
            return 4
        return 5


@cython.wraparound(False)
@cython.boundscheck(False)
def create_cell_node_map(coarse, fine, np.ndarray[PetscInt, ndim=2, mode="c"] c2f,
                         np.ndarray[PetscInt, ndim=2, mode="c"] vertex_perm):
    cdef:
        np.ndarray[PetscInt, ndim=1, mode="c"] indices, cell_map
        np.ndarray[PetscInt, ndim=2, mode="c"] permutations
        np.ndarray[PetscInt, ndim=2, mode="c"] new_cell_map, old_cell_map
        PetscInt ccell, fcell, ncoarse, ndof, i, j, perm, nfdof

    ncoarse = coarse.mesh().cell_set.size
    ndof = coarse.cell_node_map().arity

    perms = mgutils.get_node_permutations(coarse.fiat_element)
    permutations = np.empty((len(perms), len(perms.values()[0])), dtype=np.int32)
    for k, v in perms.iteritems():
        p0, p1 = np.asarray(k, dtype=PETSc.IntType)[0:2]
        permutations[hash_perm(p0, p1), :] = v[:]

    old_cell_map = fine.cell_node_map().values[c2f, ...].reshape(ncoarse, -1)

    # We're going to uniquify the maps we get out, so the first step
    # is to apply the permutation to one entry to find out which
    # indices we need to keep.
    cell_nodes = old_cell_map[0, :]
    order = -np.ones_like(cell_nodes)

    for i in range(4):
        p = permutations[hash_perm(vertex_perm[0, i*3], vertex_perm[0, i*3 + 1]), :]
        order[i*ndof:(i+1)*ndof] = cell_nodes[i*ndof:(i+1)*ndof][p]

    indices = np.empty(len(np.unique(order)), dtype=PETSc.IntType)
    seen = set()
    i = 0
    for j, n in enumerate(order):
        if n not in seen:
            indices[i] = j
            i += 1
            seen.add(n)

    nfdof = indices.shape[0]
    new_cell_map = -np.ones((ncoarse, nfdof), dtype=PETSc.IntType)

    cell_map = np.empty(4*ndof, dtype=PETSc.IntType)
    for ccell in range(ncoarse):
        # 4 fine cells per coarse
        for fcell in range(4):
            perm = hash_perm(vertex_perm[ccell, fcell*3],
                             vertex_perm[ccell, fcell*3 + 1])
            for j in range(ndof):
                cell_map[fcell*ndof + j] = old_cell_map[ccell, fcell*ndof +
                                                        permutations[perm, j]]
        for j in range(nfdof):
            new_cell_map[ccell, j] = cell_map[indices[j]]
    return new_cell_map


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
