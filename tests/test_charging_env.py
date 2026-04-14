import unittest

import numpy as np
from gymnasium import spaces

from envs.charging_env import MultiStationChargingEnv, Vehicle
from envs.maskable_actions import (
    encode_maskable_action,
    iter_valid_maskable_actions,
    no_split_action_int,
)


class ChargingEnvTests(unittest.TestCase):
    def test_reset_accepts_non_seven_station_capacities(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0, 2], duration=10.0),
            ],
            station_capacities=[1, 1, 1],
            n_bins=5,
            min_first_charge=2.0,
            min_second_charge=2.0,
        )

        observation, info = env.reset()
        expected = np.zeros(env.action_space.n, dtype=np.int8)
        for action in iter_valid_maskable_actions(
            route=[0, 2],
            n_bins=5,
            total_duration=10.0,
            t_first_min=2.0,
            t_second_min=2.0,
            num_stations=3,
        ):
            expected[action] = 1

        self.assertIsInstance(env.action_space, spaces.Discrete)
        self.assertEqual(env.action_space.n, 20)
        self.assertEqual(observation["sim_state"]["metrics"]["queue_time"], [0.0] * 3)
        np.testing.assert_array_equal(env.action_masks(), expected)
        self.assertEqual(info["total_wait"], 0.0)

    def test_reset_returns_orchestrator_observation_and_maskable_action_space(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0, 2], duration=10.0),
            ],
            station_capacities=[1, 1, 1, 1, 1, 1, 1],
            travel_time_fn=lambda a, b, _vehicle: {
                (0, 2): 3.0,
            }.get((a, b), 0.0),
            n_bins=5,
            min_first_charge=2.0,
            min_second_charge=2.0,
        )

        observation, info = env.reset()

        self.assertIsInstance(env.action_space, spaces.Discrete)
        self.assertEqual(env.action_space.n, 40)
        self.assertEqual(
            set(observation.keys()),
            {
                "sim_state",
                "commitment_features",
                "current_ev",
                "future_demand",
            },
        )
        self.assertEqual(observation["current_ev"]["station_id"], 0)
        self.assertEqual(observation["current_ev"]["total_charge_demand"], 10.0)
        self.assertEqual(observation["current_ev"]["downstream_stations"], (2,))
        self.assertEqual(observation["sim_state"]["clock"], 0.0)
        self.assertEqual(observation["sim_state"]["metrics"]["queue_time"], [0.0] * 7)
        station_payload = observation["sim_state"]["stations"][0]
        self.assertEqual(set(station_payload.keys()), {"charger_status", "queue_waiting_time", "queue_demand"})
        self.assertEqual(info["total_wait"], 0.0)

    def test_action_masks_match_maskable_action_enumerator(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0, 2, 4], duration=10.0),
            ],
            station_capacities=[1, 1, 1, 1, 1, 1, 1],
            n_bins=5,
            min_first_charge=2.0,
            min_second_charge=2.0,
        )

        env.reset()
        expected = np.zeros(env.action_space.n, dtype=np.int8)
        for action in iter_valid_maskable_actions(
            route=[0, 2, 4],
            n_bins=5,
            total_duration=10.0,
            t_first_min=2.0,
            t_second_min=2.0,
        ):
            expected[action] = 1

        np.testing.assert_array_equal(env.action_masks(), expected)

    def test_default_minimum_charging_time_masks_all_split_actions_when_demand_is_too_small(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0, 2], duration=15.0),
            ],
            station_capacities=[1, 1, 1],
            n_bins=5,
        )

        env.reset()
        expected = np.zeros(env.action_space.n, dtype=np.int8)
        expected[no_split_action_int(n_bins=5, num_stations=3)] = 1

        np.testing.assert_array_equal(env.action_masks(), expected)

    def test_step_reward_uses_delta_system_queue_time(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0], duration=5.0),
                Vehicle(vid=2, arrival_time=1.0, route=[0], duration=4.0),
            ],
            station_capacities=[1, 1, 1, 1, 1, 1, 1],
            n_bins=5,
            reward_scale=1.0,
        )

        env.reset()

        observation, reward, terminated, truncated, info = env.step(no_split_action_int(n_bins=5))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["queue_time_delta"], 0.0)
        self.assertNotIn("reward_mode", info)
        self.assertEqual(observation["current_ev"]["station_id"], 0)
        self.assertEqual(observation["current_ev"]["total_charge_demand"], 4.0)

        _observation, reward, terminated, truncated, info = env.step(no_split_action_int(n_bins=5))
        self.assertEqual(reward, -4.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["queue_time_delta"], 4.0)
        self.assertNotIn("reward_mode", info)
        self.assertEqual(_observation["current_ev"]["station_id"], 0)
        self.assertEqual(_observation["current_ev"]["total_charge_demand"], 0.0)
        self.assertEqual(info["total_wait"], 4.0)

    def test_split_action_submits_second_leg_arrival_without_another_agent_decision(self) -> None:
        env = MultiStationChargingEnv(
            vehicles=[
                Vehicle(vid=1, arrival_time=0.0, route=[0, 2], duration=20.0),
            ],
            station_capacities=[1, 1, 1],
            travel_time_fn=lambda a, b, _vehicle: {
                (0, 2): 3.0,
            }.get((a, b), 0.0),
            n_bins=5,
            min_first_charge=10.0,
            min_second_charge=10.0,
            second_leg_arrival_noise_scale=0.0,
        )

        env.reset()
        split_action = encode_maskable_action(
            second_choice=2,
            frac_bin=2,
            n_bins=5,
            num_stations=3,
        )

        observation, reward, terminated, truncated, info = env.step(split_action)

        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation["sim_state"]["metrics"]["ev_served"], [1, 0, 1])
        self.assertEqual(observation["commitment_features"]["commitment_count"], [0, 0, 0])
        self.assertEqual(info["total_wait"], 0.0)

    def test_env_no_longer_accepts_reward_mode_argument(self) -> None:
        with self.assertRaises(TypeError):
            MultiStationChargingEnv(
                vehicles=[
                    Vehicle(vid=1, arrival_time=0.0, route=[0], duration=5.0),
                ],
                station_capacities=[1, 1, 1, 1, 1, 1, 1],
                reward_mode="global",
            )


if __name__ == "__main__":
    unittest.main()
