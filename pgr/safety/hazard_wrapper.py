"""
DMC environment wrapper that adds a binary hazard cost signal.

Supports multiple environments and multi-dimensional constraints.

Environments:
  cheetah-run-v0: velocity constraint (|v| > v_max)
  walker-walk-v0: velocity + torso height constraints

Tuned so ~1-5% of timesteps incur cost for a learning agent.
"""

import gym
import numpy as np


# ── Per-environment constraint configs ────────────────────────────────────────
# Each entry: list of (obs_index, threshold, direction, name)
#   direction: 'above' means cost=1 if obs[idx] > threshold
#              'below' means cost=1 if obs[idx] < threshold

ENV_CONFIGS = {
    'cheetah-run-v0': {
        # Flat obs: 17-dim. [pos(8), vel(9)]. Forward vel = obs[8] = qvel[0].
        'constraints': [
            {'idx': 8, 'threshold': 7.0, 'direction': 'above_abs', 'name': 'velocity'},
        ],
        'obs_dim': 17,
    },
    'walker-walk-v0': {
        # Flat obs: 24-dim. [orientations(14), height(1), velocity(9)].
        # Forward vel = obs[15] = qvel[0]. Torso height = obs[14].
        'constraints': [
            {'idx': 15, 'threshold': 3.0, 'direction': 'above_abs', 'name': 'velocity'},
            {'idx': 14, 'threshold': 1.0, 'direction': 'below', 'name': 'torso_height'},
        ],
        'obs_dim': 24,
    },
}


class HazardWrapper(gym.Wrapper):
    """
    Wraps a Gym environment to add a binary cost signal.

    Supports single or multi-dimensional constraints. Cost = 1 if ANY
    constraint is violated. The cost is stored in info['cost'].

    Args:
        env: base gym environment (already wrapped by dmcgym)
        velocity_threshold: override threshold for the first (velocity) constraint.
            If None, uses ENV_CONFIGS default.
        velocity_idx: override obs index for velocity. If None, auto-detect.
        env_name: environment name for auto-config lookup.
        height_threshold: override threshold for height constraint (walker).
    """

    def __init__(self, env, velocity_threshold=None, velocity_idx=None,
                 env_name=None, height_threshold=None):
        super().__init__(env)

        # Detect environment
        self.env_name = env_name
        if env_name is None:
            # Try to detect from spec
            if hasattr(env, 'spec') and env.spec is not None:
                self.env_name = env.spec.id
            else:
                self.env_name = 'cheetah-run-v0'  # default

        # Load config
        config = ENV_CONFIGS.get(self.env_name, ENV_CONFIGS['cheetah-run-v0'])
        self.constraints = []
        for c in config['constraints']:
            constraint = dict(c)  # copy
            # Apply overrides
            if constraint['name'] == 'velocity':
                if velocity_idx is not None:
                    constraint['idx'] = velocity_idx
                if velocity_threshold is not None:
                    constraint['threshold'] = velocity_threshold
            elif constraint['name'] == 'torso_height':
                if height_threshold is not None:
                    constraint['threshold'] = height_threshold
            self.constraints.append(constraint)

        self.total_cost = 0.0
        self.episode_cost = 0.0
        self.total_steps = 0
        self.cost_steps = 0
        self.constraint_violation_counts = {c['name']: 0 for c in self.constraints}

    def reset(self, **kwargs):
        self.episode_cost = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1

        cost, violations = self._compute_cost(obs)
        info['cost'] = cost
        info['episode_cost'] = self.episode_cost
        info['constraint_violations'] = violations

        return obs, reward, done, info

    def _compute_cost(self, obs):
        """Binary cost: 1 if ANY constraint is violated, 0 otherwise."""
        violations = {}
        any_violated = False

        for c in self.constraints:
            idx = c['idx']
            if idx >= len(obs):
                violations[c['name']] = False
                continue

            val = obs[idx]
            direction = c['direction']
            threshold = c['threshold']

            if direction == 'above':
                violated = val > threshold
            elif direction == 'above_abs':
                violated = abs(val) > threshold
            elif direction == 'below':
                violated = val < threshold
            else:
                violated = False

            violations[c['name']] = bool(violated)
            if violated:
                any_violated = True
                self.constraint_violation_counts[c['name']] += 1

        cost = 1.0 if any_violated else 0.0
        self.total_cost += cost
        self.episode_cost += cost
        if cost > 0:
            self.cost_steps += 1
        return cost, violations

    @property
    def hazard_rate(self):
        if self.total_steps == 0:
            return 0.0
        return self.cost_steps / self.total_steps

    @property
    def num_constraints(self):
        return len(self.constraints)

    def get_constraint_info(self):
        """Return a summary of active constraints for logging."""
        return {c['name']: {'idx': c['idx'], 'threshold': c['threshold'],
                            'direction': c['direction']}
                for c in self.constraints}


class TwoPhaseHazardWrapper(gym.Wrapper):
    """
    Wraps a HazardWrapper to suppress cost during a 'safe' phase.

    Phase 1 (0 to safe_start_frac):     hazards active
    Phase 2 (safe_start_frac to safe_end_frac): hazards suppressed (cost=0)
    Phase 3 (safe_end_frac to 1.0):     hazards return
    """

    def __init__(self, env, total_steps, safe_start_frac=0.3, safe_end_frac=0.7):
        super().__init__(env)
        self.total_training_steps = total_steps
        self.safe_start_step = int(total_steps * safe_start_frac)
        self.safe_end_step = int(total_steps * safe_end_frac)
        self.current_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1

        # Suppress cost during safe phase
        if self.safe_start_step <= self.current_step < self.safe_end_step:
            info['cost'] = 0.0
            info['hazards_active'] = False
        else:
            info['hazards_active'] = True

        return obs, reward, done, info
