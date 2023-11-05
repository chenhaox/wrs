from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletPlaneShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh, BulletContactResult
import numpy as np
import basis.data_adapter as da
from panda3d.core import TransformState

physicsworld = BulletWorld()

def gen_cdmesh(trm_model):
    pdgeom = da.pdgeom_from_vvnf(trm_model.vertices, trm_model.vertex_normals, trm_model.faces)
    pdbtrm = BulletTriangleMesh()
    pdbtrm.addGeom(pdgeom)
    pdbtrmshape = BulletTriangleMeshShape(pdbtrm, dynamic=True)
    pdbtrmshape.setMargin(0)
    pdbrbd_nd = BulletRigidBodyNode(name="auto")
    pdbrbd_nd.addShape(pdbtrmshape)
    return pdbrbd_nd


def update_pose(pdbrbd_nd, objcm):
    """
    update pdbrbd_nd using the Mat of objcm.pdndp
    :param pdbrbd_nd:
    :param objcm:
    :return:
    author: weiwei
    date: 20211215
    """
    pdbrbd_nd.setTransform(TransformState.makeMat(objcm.pdndp.getMat()))


def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
    """
    generate a plane bulletrigidbody node
    :param updirection: the normal parameter of bulletplaneshape at panda3d
    :param offset: the d parameter of bulletplaneshape at panda3d
    :param name:
    :return: bulletrigidbody
    author: weiwei
    date: 20170202, tsukuba
    """
    pdbrbd_nd = BulletRigidBodyNode(name)
    pdbplaneshape = BulletPlaneShape(da.npvec3_to_pdvec3(updirection), offset)
    pdbplaneshape.setMargin(0)
    pdbrbd_nd.addShape(pdbplaneshape)
    return pdbrbd_nd


def is_collided(objcm0, objcm1, toggle_contacts=True):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0:
    :param objcm1:
    :param toggle_contactpoints: True default
    :return:
    author: weiwei
    date: 20210117, 20211215
    """
    max_contacts=10
    pdbrbd_nd0 = objcm0.copy_transformed_cdmesh()
    pdbrbd_nd1 = objcm1.copy_transformed_cdmesh()
    physicsworld.attach(pdbrbd_nd0)
    physicsworld.attach(pdbrbd_nd1)
    result = physicsworld.contactTestPair(pdbrbd_nd0, pdbrbd_nd1)
    contacts = result.getContacts()
    contact_points = [da.pdvec3_to_npvec3(ct.getManifoldPoint().getPositionWorldOnB()) for ct in contacts]
    contact_points += [da.pdvec3_to_npvec3(ct.getManifoldPoint().getPositionWorldOnA()) for ct in contacts]
    contact_points = contact_points[0:max_contacts]
    physicsworld.remove(pdbrbd_nd0)
    physicsworld.remove(pdbrbd_nd1)
    if toggle_contacts:
        return (True, np.asarray(contact_points)) if len(contact_points) > 0 else (False, np.asarray(contact_points))
    else:
        return True if len(contact_points) > 0 else False


def rayhit_closet(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805, 20210118
    """
    pdbrbd_nd = objcm.copy_transformed_cdmesh()
    physicsworld.attach(pdbrbd_nd)
    result = physicsworld.rayTestClosest(da.npvec3_to_pdvec3(pfrom), da.npvec3_to_pdvec3(pto))
    physicsworld.remove(pdbrbd_nd)
    if result.hasHit():
        return [da.pdvec3_to_npvec3(result.getHitPos()), da.pdvec3_to_npvec3(result.getHitNormal())]
    else:
        return [None, None]


def rayhit_all(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805, 20210118
    """
    pdbrbd_nd = objcm.copy_transformed_cdmesh()
    physicsworld.attach(pdbrbd_nd)
    result = physicsworld.rayTestAll(da.npvec3_to_pdvec3(pfrom), da.npvec3_to_pdvec3(pto))
    physicsworld.remove(pdbrbd_nd)
    if result.hasHits():
        return [[da.pdvec3_to_npvec3(hit.getHitPos()), da.pdvec3_to_npvec3(-hit.getHitNormal())] for hit in result.getHits()]
    else:
        return []


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import basis.robot_math as rm

    # wd.World(cam_pos=[.2, -.1, .2], lookat_pos=[0, 0, 0.05])
    # gm.gen_frame().attach_to(base)
    # # objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    # # objcm1 = cm.CollisionModel(objpath, cdmesh_type='triangles')
    # objcm1 = cm.gen_stick(major_radius=.01)
    # pos = np.array([[-0.5, -0.82363909, 0.2676166, -0.00203699],
    #                     [-0.86602539, 0.47552824, -0.1545085, 0.01272306],
    #                     [0., -0.30901703, -0.95105648, 0.12604253],
    #                     [0., 0., 0., 1.]])
    # # pos = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
    # #                     [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
    # #                     [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
    # #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # objcm1.set_homomat(pos)
    # objcm1.set_rgba([1, 1, .3, .01])
    # # objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    # # objcm2 = cm.CollisionModel(objpath, cdmesh_type='triangles')
    # objcm2 = cm.gen_stick(major_radius=.01)
    # objcm2.set_rgba([1,0,1,.2])
    # objcm2.set_pos(np.array([-.018,-.018,0]))
    # iscollided, contact_points = is_collided(objcm1, objcm2)
    # objcm1.show_cdmesh()
    # objcm2.show_cdmesh()
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct_pnt in contact_points:
    #     gm.gen_sphere(ct_pnt, major_radius=.001).attach_to(base)
    # # spos = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # # hitpos, hitnrml = rayhit_closet(spos=spos, epos=epos, objcm=objcm2)
    # # gm.gen_sphere(hitpos, major_radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # # gm.gen_stick(spos=spos, epos=epos, major_radius=.002).attach_to(base)
    # # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    # base.run()

    wd.World(cam_pos=[.3, -.3, .3], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm1 = cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1, 1, .3, .2])
    objcm2 = cm.CollisionModel(objpath)
    # objcm2= cm.gen_stick(major_radius=.07)
    # objcm2.set_rgba([1, 0, 1, .1])
    objcm2.set_pos(objcm1.get_pos() + np.array([.0, .03, .0]))
    # objcm1.change_cdmesh_type('convex_hull')
    # objcm2.change_cdmesh_type('obb')
    iscollided, contact_points = is_collided(objcm1, objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        print("draw points")
        gm.gen_sphere(ctpt, radius=.01).attach_to(base)
    # spos = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # rayhit_closet(spos=spos, epos=epos, objcm=objcm2)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(end_type='box')
    # objcm.show_cdmesh(end_type='convex_hull')
    # gm.gen_sphere(hitpos, major_radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=spos, epos=epos, major_radius=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
