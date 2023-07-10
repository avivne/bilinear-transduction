import numpy as np
# import trimesh
import random
import pdb
import io
from PIL import Image, ImageOps
import copy
import os
import os.path as osp
# from sklearn.decomposition import PCA

class Grasp():
    def __init__(self, mug_args):
        # self.demos_path = mug_args['demos_path']
        # self.source_meshes = []
        # self.source_grasps = []
        # self.source_centers = []
        # for demo_path in sorted(os.listdir(self.demos_path)):
        #     grasp_demo = np.load(osp.join(self.demos_path, demo_path))
        #     source_mesh = trimesh.points.PointCloud(vertices=grasp_demo['object_pointcloud'].reshape(-1,3))
        #     source_grasp = trimesh.points.PointCloud(vertices=grasp_demo['ee_pose_world'][:3].reshape(-1,3))

        #     #normalize z axis table=0 for scale [min z]
        #     #normalize center x-y [min x-y for 2% of lowest z]
        #     bottom_mesh = trimesh.points.PointCloud(vertices=source_mesh[np.argsort(source_mesh.vertices[:,-1])[:np.int(len(source_mesh.vertices)*0.02)]])
        #     center_approx = np.array([np.mean(bottom_mesh.vertices[:,0]), np.mean(bottom_mesh.vertices[:,1]), np.min(bottom_mesh.vertices[:,2])])
        #     source_center = trimesh.points.PointCloud(vertices=center_approx.reshape(-1,3))
        #     source_mesh.apply_translation(-center_approx)
        #     source_grasp.apply_translation(-center_approx)
        #     source_center.apply_translation(-center_approx) #center in 0,0,0

        #     # print(demo_path)
        #     # print(source_mesh.vertices.shape)
        #     self.source_meshes.append(source_mesh)
        #     self.source_grasps.append(source_grasp)
        #     self.source_centers.append(source_center)

        self.n_PCA_comp = mug_args['n_PCA_comp']
        self.mesh_noise = mug_args['mesh_noise']
        self.grasp_noise = mug_args['grasp_noise']
        self.type_idxs = mug_args['type_idxs']
        self.sample_size = mug_args['sample_size']


    # def collect_data(self, n_samples, r_range=None, t_range=None, s_range=None, single_transformation=False):
    #     X = []
    #     X_params = [] #for plotting
    #     Y = []
    #     M = [] #sampled mesh for PointNet++ dataset (can reconstruct the orig mesh from source mesh and X_params)
    #     for obj_idx in range(len(self.source_meshes)): #mug, bottle, teapot
    #         for _ in range(n_samples//len(self.source_meshes)):
    #             mesh = copy.deepcopy(self.source_meshes[obj_idx])
    #             grasp = copy.deepcopy(self.source_grasps[obj_idx])
    #             center = copy.deepcopy(self.source_centers[obj_idx])
    #             # every sample may undergo any of the 3 transformations: rotation, translation, scale
    #             offset_x, offset_y, offset_z, theta, scale_factor = None, None, None, None, None
    #             r_transformation, t_transformation, s_transformation = True, True, True
    #             if single_transformation: #sample transformation and set transformation bools
    #                 transformations = [False, False, False]
    #                 transformations[np.random.randint(3, size=1)[0]] = True
    #                 r_transformation, t_transformation, s_transformation = transformations
    #             #THE ORDER OF TRANSFORMATIONS MATTERS!
    #             #rotation
    #             if r_range is not None and r_transformation:
    #                 rot = np.eye(4)
    #                 theta = random.uniform(r_range[0], r_range[1]) * (2*np.pi)
    #                 rot[0,0] = rot[1,1] = np.cos(theta); rot[0,1] = np.sin(theta); rot[1,0] = -np.sin(theta)
    #                 mesh.apply_transform(rot)
    #                 grasp.apply_transform(rot)
    #                 center.apply_transform(rot)
    #             #translation
    #             if t_range is not None and t_transformation:
    #                 offset_x, offset_y = np.random.uniform(t_range[0], t_range[1], 2)
    #                 offset_z = 0
    #                 mesh.apply_translation([offset_x, offset_y, offset_z])
    #                 grasp.apply_translation([offset_x, offset_y, offset_z])
    #                 center.apply_translation([offset_x, offset_y, offset_z])
    #             #scale
    #             if s_range is not None and s_transformation:
    #                 scale_factor = random.uniform(s_range[0], s_range[1])
    #                 mesh.apply_scale(scale_factor)
    #                 grasp.apply_scale(scale_factor)
    #                 center.apply_scale(scale_factor)

    #             #noise mesh
    #             mu, sigma = 0, self.mesh_noise
    #             mesh_vertices = np.array(mesh.vertices)
    #             mesh_shape = mesh_vertices.shape
    #             noise = np.random.normal(mu, sigma, [mesh_shape[0], mesh_shape[1]])
    #             mesh = trimesh.points.PointCloud(vertices=mesh_vertices+noise)
    #             # noise grasping
    #             mu, sigma = 0, self.grasp_noise
    #             grasp_gauss_noise = np.random.normal(mu, sigma, [1, 3])[0].tolist()
    #             grasp.apply_translation(grasp_gauss_noise)

    #             # [x, y, z, theta, alpha]
    #             params_state = [offset_x, offset_y, offset_z, theta, scale_factor]

    #             # [x, y, z] mean
    #             state = list(np.array(mesh.centroid)) #point cloud mean, mesh.center_mass not defined (pointcloud not watertight)
    #             # n_components directions which explain most variance (max 3)
    #             pca = PCA(n_components=self.n_PCA_comp)
    #             pca.fit(np.array(mesh.vertices))
    #             state += list(pca.components_.flatten())
    #             #one-hot info re object type for priviledged training with known weights
    #             object_one_hot = [0 for _ in range(len(self.source_meshes))]
    #             object_one_hot[obj_idx] = 1
    #             state += object_one_hot

    #             sampled_mesh = mesh.vertices[random.sample(range(mesh.vertices.shape[0]), self.sample_size)]

    #             X.append(state)
    #             X_params.append(params_state)
    #             Y.append(grasp.vertices.tolist()[0])
    #             M.append(sampled_mesh)

    #     return np.array(X), np.array(X_params), np.array(Y), np.array(M)


    # def transform_mesh(self, x_state, x_params):
    #     offset_x, offset_y, offset_z, theta, scale_factor = x_params
    #     #mesh
    #     obj_idx = np.where(x_state[self.type_idxs])[0][0]
    #     mesh = copy.deepcopy(self.source_meshes[obj_idx])
    #     #sparse mesh
    #     sampled_mesh = mesh.vertices[random.sample(range(mesh.vertices.shape[0]), 3000)]
    #     mesh = trimesh.points.PointCloud(vertices=sampled_mesh)
    #     #rotation
    #     if theta is not None:
    #         rot = np.eye(4)
    #         rot[0,0] = rot[1,1] = np.cos(theta); rot[0,1] = np.sin(theta); rot[1,0] = -np.sin(theta)
    #         mesh.apply_transform(rot)
    #     #translation
    #     if offset_x is not None and offset_y is not None and offset_z is not None:
    #         mesh.apply_translation([offset_x, offset_y, offset_z])
    #     #scale
    #     if scale_factor is not None:
    #         mesh.apply_scale(scale_factor)
        
    #     return mesh


    # def plot_pcd(self, x_state, x_params, grasp_pred, grasp_gt, save_path, a_state=None, a_params=None):
    # # def plot_mug(self, x_M, grasp_pred, grasp_gt, save_path, a_M):
    #     # run with xvfb-run -a python *.py
    #     #grasp
    #     # grasp_pred = trimesh.points.PointCloud(vertices=grasp_pred.reshape(-1,3), colors=np.array([255, 0, 0, 255], dtype=np.uint8)) #red
    #     # grasp_gt = trimesh.points.PointCloud(vertices=grasp_gt.reshape(-1,3), colors=np.array([50, 0, 255, 255], dtype=np.uint8)) #blue
    #     distance = 0.5
    #     if x_state[self.type_idxs][-1]: distance = 0.7 #teapot
    #     lambda_sphere = 150
    #     pred_sphere = np.random.uniform(-1,1,(100000,3))
    #     pred_sphere /= lambda_sphere * np.linalg.norm(pred_sphere, axis=1).reshape((-1,1))
    #     pred_sphere += grasp_pred
    #     grasp_pred = trimesh.points.PointCloud(vertices=pred_sphere, colors=np.array([231, 76, 60, 255], dtype=np.uint8)) #red [rgb(231, 76, 60)] [255, 0, 0, 255]  
    #     gt_sphere = np.random.uniform(-1,1,(100000,3))
    #     gt_sphere /= lambda_sphere * np.linalg.norm(gt_sphere, axis=1).reshape((-1,1))
    #     gt_sphere += grasp_gt
    #     grasp_gt = trimesh.points.PointCloud(vertices=gt_sphere, colors=np.array([46, 204, 113, 255], dtype=np.uint8)) #green rgb(46, 204, 113)  blue [52, 152, 219] [50, 0, 255, 255]
                
    #     mesh = self.transform_mesh(x_state, x_params) #black
    #     # mesh = trimesh.points.PointCloud(vertices=x_M)
    #     plot_pcds = [mesh, grasp_pred, grasp_gt]
    #     # if a_state is not None and a_params is not None: #anchor
    #     # if a_M is not None: #anchor
    #     if a_state is not None:
    #         # anchor_mesh = self.transform_mesh(a_state, a_params)
    #         a_M = self.transform_mesh(a_state, a_params).vertices
    #         anchor_mesh = trimesh.points.PointCloud(vertices=a_M, colors=np.array([149, 165, 166, 255], dtype=np.uint8)) #green
    #         # plot_pcds += [anchor_mesh]
    #     for thetay in [45,60,75,90,105,120,135,150,165,180,195]:
    #         # test sample
    #         scene = trimesh.Scene()
    #         scene.add_geometry(plot_pcds)
    #         rotx = np.eye(4); roty = np.eye(4)
    #         thetax = 180
    #         rotx[1,1] = rotx[2,2] = np.cos(thetax); rotx[1,2] = np.sin(thetax); rotx[2,1] = -np.sin(thetax) #x axis
    #         roty[0,0] = roty[2,2] = np.cos(thetay); roty[0,2] = -np.sin(thetay); roty[2,0] = np.sin(thetay) #y axis
    #         rot = np.matmul(rotx, roty)  # rot = np.matmul(np.matmul(rotx, roty), rotz)
    #         points = np.concatenate([np.array(grasp_pred.vertices), np.array(grasp_gt.vertices)], axis=0)
    #         # # if a_M is not None: #anchor
    #         # if a_state is not None:
    #         #     # points = np.concatenate([points, np.mean(anchor_mesh.vertices, axis=0).reshape(-1,3)])
    #         #     points = np.concatenate([points, anchor_mesh.vertices])
    #         points = np.concatenate([points, mesh.vertices])
    #         camera_transformation = scene.camera.look_at(points=points, rotation=rot, \
    #                                                      center=np.mean(mesh.vertices,axis=0), distance=distance)
    #         # camera_transformation = scene.camera.look_at(points=points, rotation=rot, \
    #         #                                              center=np.mean(grasp_gt.vertices,axis=0), distance=1)
    #         scene.camera_transform = camera_transformation
    #         data = scene.save_image(resolution=(1080,1080)) #, visible=False
    #         image = Image.open(io.BytesIO(data))
    #         gray_image = ImageOps.grayscale(image)
    #         image.save(f'{save_path}_{str(thetay)}.png', bbox_inches='tight')

    #         #anchor
    #         # if a_M is not None: #anchor
    #         if a_state is not None:            
    #             scene = trimesh.Scene()
    #             scene.add_geometry([anchor_mesh])
    #             points = anchor_mesh.vertices
    #             camera_transformation = scene.camera.look_at(points=points, rotation=rot, \
    #                                                          center=np.mean(anchor_mesh.vertices,axis=0), distance=distance)
    #             scene.camera_transform = camera_transformation
    #             data = scene.save_image(resolution=(1080,1080)) #, visible=False
    #             image = Image.open(io.BytesIO(data))
    #             image.save(f'{save_path}_{str(thetay)}_anchor.png', bbox_inches='tight')
