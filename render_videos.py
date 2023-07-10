import pdb
import os
import cv2
import numpy as np
import os
import os.path as osp
from PIL import Image
import PIL.ImageDraw as ImageDraw
from ruamel.yaml import YAML
from utils.util import make_env, load_pkl


def writer_for(log_dir, tag, fps, res):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return cv2.VideoWriter(
        f'{log_dir}/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )


def label_with_model_dist(frame, tag):
    tag = tag.replace('demos', 'Demos')
    tag = tag.replace('bc', 'NN')
    tag = tag.replace('biLinear', 'Bilinear Transduction')

    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    #EXPERT, BC: in distributiom, BC: out-of-support....
    drawer.text((im.size[0]/20,im.size[1]/18), tag, fill=text_color)
    return np.array(im)


def trajectory2vid(env, traj, tag, env_name, goal_idxs, resolution, flip=False):  
    print(env_name, tag)   
    imgs = []  
    env.reset()
    if env_name in ['reach-v2', 'push-v2']: #metaworld
        goal_pos = traj['obs'][-1, goal_idxs]
        o = env.reset_model_ood(goal_pos=goal_pos)
        for st in range(len(traj['action'])):
            no, r, _, info = env.step(traj['action'][st])
            # camera in ['corner', 'topview', 'behindGripper', 'gripperPOV']
            img = env.sim.render(*resolution, mode='offscreen', camera_name='corner')[:,:,::-1]
            if flip: img = cv2.rotate(img, cv2.ROTATE_180) # if True, flips output image 180 degrees
            img = label_with_model_dist(img, tag)
            imgs.append(img)      
    elif env_name in ['slider']:
        mass = traj['obs'][-1][goal_idxs][0]
        o = env.set_goal(mass)
        for st in range(len(traj['action'])):
            #TODO does step get same obs?
            no, r, _, info = env.step(traj['action'][st])
            # img = env.sim.render(480, 640, mode='offscreen', camera_name='topview')
            img = env.sim.render(*resolution, mode='offscreen', camera_name='topview')
            img = label_with_model_dist(img, tag) #TODO array vs image   
            imgs.append(img)                       
    elif env_name in ['adroit', 'adroit_scale']:
        # img = env.env.env.env.sim.render(640, 480, mode='offscreen')
        goal_pos = traj['obs'][-1, goal_idxs]
        o = env.set_goal(goal_pos=goal_pos)
        for st in range(len(traj['action'])):
            no, r, _, info = env.step(traj['action'][st])
            img = env.env.env.env.sim.render(*resolution, mode='offscreen')
            img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = label_with_model_dist(img, tag)            
            imgs.append(img)          
    return imgs


def get_end_dist(traj, eval_idxs, target):
    return round(np.linalg.norm(traj['obs'][-1][eval_idxs] - target),3)


def get_mass(traj, goal_idxs):
    return round(traj['obs'][-1][goal_idxs][0])


config_base_dir = 'configs'
demo_base_dir = 'data'
exper_base_dir = 'log'

model_types = {}
####################################################
# MODIFY to generate videos of interest
config_names = ['adroit.yaml', 'push_metaworld.yaml', 'reach_metaworld.yaml', 'slider.yaml']
expert_idxs = [0] 
mode_eval_idxs = {'in_dist': [0], 'ood': [0]}
model_types['adroit'] = ['bc_lsize512_lnum1_embed32', 'bilinear_lsize512_lnum1_embed32']
model_types['push-v2'] = ['bc_lsize32_lnum1_embed32', 'bilinear_lsize32_lnum1_embed32']
model_types['reach-v2'] = ['bc_lsize32_lnum1_embed32', 'bilinear_lsize32_lnum1_embed32']
model_types['slider'] = ['bc_lsize32_lnum1_embed32', 'bilinear_lsize32_lnum1_embed32']
####################################################


resolution = (560, 420)
for config_name in config_names:
    config_name = osp.join(config_base_dir, config_name)
    yaml = YAML()
    v = yaml.load(open(config_name))
    env_name = v['env']['env_name']
    vidpath = env_name
    env = make_env(env_name, v['env'])
    goal_idxs = v['env']['goal_idxs']  
    eval_idxs = v['env']['plot_idxs']
    imgs = []
    demos = load_pkl(osp.join(demo_base_dir, env_name, 'demos.pkl'))
    for idx in expert_idxs:
        traj = demos[idx]
        if env_name in ['slider']:
            mass = get_mass(traj, goal_idxs)
            end_dist = get_end_dist(traj, eval_idxs, env.goal)
            curr_tag = f'demos mass {mass}: final distance {end_dist}'
        else:
            end_dist = get_end_dist(traj, eval_idxs, traj['obs'][-1][goal_idxs])
            curr_tag = f'demos: final distance {end_dist}'   
        imgs += trajectory2vid(env, traj, curr_tag, env_name, goal_idxs, resolution)    
    for model_type in model_types[env_name]:
        tag = model_type.split('_')[0]
        for mode in ['in_dist', 'ood']:
            model_demos = load_pkl(osp.join(exper_base_dir, env_name, model_type, f'{tag}_eval_{mode}.pkl'))
            for idx in mode_eval_idxs[mode]:
                tag = model_type.split('_')[0] #frame title EXPERT, BC: in distributiom, BC: out-of-support....
                traj = model_demos[idx]                
                if env_name in ['slider']:
                    mass = get_mass(traj, goal_idxs)
                    end_dist = get_end_dist(traj, eval_idxs, env.goal)
                    curr_tag = f'{tag} {mode}: mass {mass}, final distance {end_dist}'
                else:
                    end_dist = get_end_dist(traj, eval_idxs, traj['obs'][-1][goal_idxs]) #good for adroit
                    curr_tag = f'{tag} {mode}: final distance {end_dist}'
                # else:
                #     end_dist = get_end_dist(traj, eval_idxs, traj['obs'][-1][goal_idxs][0]) #TODO
                #     curr_tag = f'{tag} {mode}: final distance {end_dist}'                    
                imgs += trajectory2vid(env, traj, curr_tag, env_name, goal_idxs, resolution)
    fps = 75 #env.metadata['video.frames_per_second']
    vid_path = env_name
    writer = writer_for('videos', vidpath, fps, resolution)
    for img in imgs:
        writer.write(img) 
    writer.release()
    cv2.destroyAllWindows()
