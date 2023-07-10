from collections import OrderedDict
import re

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerPushEnvV2,
    SawyerReachEnvV2,
)


ALL_V2_ENVIRONMENTS = OrderedDict((
    ('push-v2', SawyerPushEnvV2),
    ('reach-v2', SawyerReachEnvV2),
))



def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                np.random.set_state(st0)

        d['__init__'] = initialize
        d['__module__'] = __name__
        og_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        og_env_name = og_env_name.replace("-", "")

        og_env_key = '{}-goal-observable'.format(env_name)
        og_env_name = '{}GoalObservable'.format(og_env_name)
        ObservableGoalEnvCls = type(og_env_name, (env_cls, ), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()
