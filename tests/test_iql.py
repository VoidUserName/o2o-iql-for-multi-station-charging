import unittest

import numpy as np

from envs.charging_env import Vehicle
from envs.maskable_actions import encode_maskable_action, no_split_action_int
from train.iql.agent import DiscreteIQLAgent, sanitize_action_masks
from train.iql.data import collect_expert_transitions


class IQLDataTests(unittest.TestCase):
    def test_collect_expert_transitions_rolls_each_pair_in_its_own_episode_env(self) -> None:
        episode_one = [
            Vehicle(vid=1, arrival_time=0.0, route=[0], duration=20.0),
        ]
        episode_two = [
            Vehicle(vid=2, arrival_time=0.0, route=[0, 1], duration=30.0),
        ]
        paired_data = [
            (episode_one, {}),
            (episode_two, {2: {"vehicle_id": "2", "station_0": "15", "station_1": "15"}}),
        ]

        dataset = collect_expert_transitions(
            paired_data=paired_data,
            n_bins=21,
            max_queue_len=4,
            seed=123,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.actions[0], no_split_action_int(n_bins=21, num_stations=7))
        self.assertEqual(
            dataset.actions[1],
            encode_maskable_action(second_choice=1, frac_bin=10, n_bins=21, num_stations=7),
        )
        self.assertTrue(np.all(dataset.action_masks[np.arange(2), dataset.actions] == 1))


class IQLAgentTests(unittest.TestCase):
    def test_sanitize_action_masks_falls_back_to_all_valid(self) -> None:
        masks = sanitize_action_masks(
            action_masks=np.array([[0, 0, 0], [1, 0, 1]], dtype=np.uint8),
            action_dim=3,
        )
        self.assertTrue(bool(masks[0].all()))
        self.assertTrue(bool(masks[1, 0]))
        self.assertTrue(bool(masks[1, 2]))

    def test_agent_act_never_returns_masked_action(self) -> None:
        agent = DiscreteIQLAgent(
            obs_dim=5,
            act_dim=4,
            hidden_dims=(16, 16),
            device="cpu",
        )
        observation = np.zeros(5, dtype=np.float32)
        mask = np.array([0, 1, 0, 0], dtype=np.uint8)

        deterministic_action = agent.act(observation, mask, deterministic=True)
        sampled_actions = {agent.act(observation, mask, deterministic=False) for _ in range(20)}

        self.assertEqual(deterministic_action, 1)
        self.assertEqual(sampled_actions, {1})


if __name__ == "__main__":
    unittest.main()
