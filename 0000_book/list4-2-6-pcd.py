import time
import numpy as np
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    # ウサギのモデルのファイルを用いてCollisionModelを初期化します
    # ウサギ1~5はこのCollisionModelのコピーとして定義します
    objcm_ref = cm.CollisionModel(initializer="./objects/bunnysim.stl",
                                  cdprimitive_type=cm.CDPrimitiveType.BOX,
                                  cdmesh_type=cm.CDMeshType.DEFAULT)
    objcm_ref.set_rgba([.9, .75, .35, 1])
    objcm1 = objcm_ref.copy()
    objcm1.set_pos(np.array([0, -.01, 0]))
    objcm1.attach_to(base)
    objcm1.show_cdprimitive()
    objcm2 = objcm_ref.copy()
    objcm2.set_pos(np.array([0, .03, 0]))
    objcm2.attach_to(base)
    objcm2.show_cdprimitive()
    objcm2.change_cdprimitive_type(cdprimitive_type=cm.CDPrimitiveType.CYLINDER)
    tic = time.time()
    result, contact_points = objcm1.is_pcdwith(objcm2, toggle_contacts=True)
    toc = time.time()
    print(f"Time cost is: {toc - tic}", result)
    if result:
        for point in contact_points:
            gm.gen_sphere(point).attach_to(base)
    base.run()
