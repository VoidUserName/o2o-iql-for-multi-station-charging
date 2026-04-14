import unittest

import numpy as np

from train.iql.data import TransitionDataset
from train.o2o_iql.replay import BalancedReplayManager, DensityRatioEstimator


def _make_dataset(observations: np.ndarray, actions: np.ndarray, act_dim: int) -> TransitionDataset:
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    action_masks = np.ones((len(observations), act_dim), dtype=np.uint8)
    return TransitionDataset(
        observations=observations,
        actions=actions,
        rewards=np.zeros(len(observations), dtype=np.float32),
        next_observations=observations.copy(),
        dones=np.zeros(len(observations), dtype=np.float32),
        action_masks=action_masks,
    )


class BalancedReplayTests(unittest.TestCase):
    def test_manager_uses_offline_until_online_buffer_is_warm(self) -> None:
        rng = np.random.default_rng(0)
        offline_dataset = _make_dataset(
            observations=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            actions=np.array([0, 0, 1, 1], dtype=np.int64),
            act_dim=2,
        )
        manager = BalancedReplayManager(
            offline_dataset=offline_dataset,
            obs_dim=1,
            act_dim=2,
            online_buffer_size=8,
            online_sample_prob=1.0,
            min_online_samples=3,
            priority_refresh_freq=10,
            priority_model_steps=1,
            priority_batch_size=4,
            device="cpu",
        )

        batch, source = manager.sample(batch_size=2, rng=rng)
        self.assertEqual(source, "offline")
        self.assertEqual(batch["observations"].shape, (2, 1))

        for value in [10.0, 11.0]:
            manager.add_online_transition(
                observation=np.array([value], dtype=np.float32),
                action=1,
                reward=0.0,
                next_observation=np.array([value], dtype=np.float32),
                done=False,
                action_mask=np.array([1, 1], dtype=np.uint8),
            )

        _, source = manager.sample(batch_size=2, rng=rng)
        self.assertEqual(source, "offline")

        manager.add_online_transition(
            observation=np.array([12.0], dtype=np.float32),
            action=1,
            reward=0.0,
            next_observation=np.array([12.0], dtype=np.float32),
            done=False,
            action_mask=np.array([1, 1], dtype=np.uint8),
        )
        _, source = manager.sample(batch_size=2, rng=rng)
        self.assertEqual(source, "online")

    def test_density_ratio_scores_online_like_offline_samples_higher(self) -> None:
        rng = np.random.default_rng(1)
        offline_dataset = _make_dataset(
            observations=np.array(
                [[0.0], [0.2], [0.4], [0.6], [8.0], [8.2], [8.4], [8.6]],
                dtype=np.float32,
            ),
            actions=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
            act_dim=2,
        )
        manager = BalancedReplayManager(
            offline_dataset=offline_dataset,
            obs_dim=1,
            act_dim=2,
            online_buffer_size=16,
            online_sample_prob=0.5,
            min_online_samples=2,
            priority_refresh_freq=5,
            priority_model_steps=200,
            priority_batch_size=8,
            priority_model_lr=5e-3,
            priority_uniform_floor=0.01,
            device="cpu",
        )

        for value in [0.05, 0.15, 0.25, 0.35]:
            manager.add_online_transition(
                observation=np.array([value], dtype=np.float32),
                action=0,
                reward=0.0,
                next_observation=np.array([value], dtype=np.float32),
                done=False,
                action_mask=np.array([1, 1], dtype=np.uint8),
            )

        stats = manager.maybe_refresh_priorities(step=5, rng=rng)
        self.assertIsNotNone(stats)
        priorities = manager.offline_buffer.priorities
        self.assertGreater(float(priorities[:4].mean()), float(priorities[4:].mean()))
        self.assertAlmostEqual(float(priorities.sum()), 1.0, places=6)


class DensityEstimatorTests(unittest.TestCase):
    def test_density_estimator_scores_separable_classes(self) -> None:
        rng = np.random.default_rng(2)
        offline_dataset = _make_dataset(
            observations=np.array([[-2.0], [-1.5], [5.0], [5.5]], dtype=np.float32),
            actions=np.array([0, 0, 1, 1], dtype=np.int64),
            act_dim=2,
        )
        estimator = DensityRatioEstimator(
            obs_dim=1,
            act_dim=2,
            hidden_dims=(32, 32),
            learning_rate=5e-3,
            device="cpu",
        )

        online_manager = BalancedReplayManager(
            offline_dataset=offline_dataset,
            obs_dim=1,
            act_dim=2,
            online_buffer_size=8,
            online_sample_prob=0.5,
            min_online_samples=1,
            priority_refresh_freq=10,
            priority_model_steps=1,
            priority_batch_size=4,
            device="cpu",
        )
        for value in [-1.9, -1.7, -1.4]:
            online_manager.add_online_transition(
                observation=np.array([value], dtype=np.float32),
                action=0,
                reward=0.0,
                next_observation=np.array([value], dtype=np.float32),
                done=False,
                action_mask=np.array([1, 1], dtype=np.uint8),
            )

        loss = estimator.fit(
            offline_dataset=offline_dataset,
            online_buffer=online_manager.online_buffer,
            rng=rng,
            n_steps=150,
            batch_size=6,
        )
        scores = estimator.score_offline_dataset(offline_dataset)

        self.assertGreater(loss, 0.0)
        self.assertGreater(float(scores[:2].mean()), float(scores[2:].mean()))


if __name__ == "__main__":
    unittest.main()
