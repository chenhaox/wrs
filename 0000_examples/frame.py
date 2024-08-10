"""
Created on 2024/6/17 
Author: Hao Chen (chen960216@gmail.com)
"""
import numpy as np
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd

base = wd.World(cam_pos=[1, .7, .3], lookat_pos=[0, 0, 0])
# pos = [-0.13579653 -0.02226533  0.44797869]
# rot  = [[ 0.98029242 -0.01339614 -0.19709722]
#  [-0.07744752  0.8917719  -0.44580799]
#  [ 0.18173787  0.45228688  0.87315974]]
rot = np.array([[0.98029242, -0.01339614, -0.19709722],
                [-0.07744752, 0.8917719, -0.44580799],
                [0.18173787, 0.45228688, 0.87315974]])

gm.gen_frame(rotmat=rot).attach_to(base)
base.run()