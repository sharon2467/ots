"""Regression checks for configurable sampler, layout and output blocks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import data
import main
import model
import train
from tests.test_model_train_t3 import valid_checkpoint_payload


class TestSecondDataset(unittest.TestCase):
    def test_layout_limits_and_partial_override(self):
        config = data._physical_config_fig1_fig11(fourier_modes=1001, used_in_model={"e": False})
        self.assertEqual(config["fourier_modes"], 1001)
        self.assertFalse(config["electron_dis"]["used_in_model"])
        self.assertTrue(config["ion1_dis"]["used_in_model"])
        self.assertEqual(len(config["ion1_dis"]["velocity_grid_m_s"]), 4097)
        for overrides in ({"used_in_model": {"wrong": True}},
                          {"velocity_grids_m_s": {"wrong": [1, 2, 3]}},
                          {"used_in_model": {"e": False, "i1": False, "i2": False}},
                          {"fourier_modes": 4098}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                data._physical_config_fig1_fig11(**overrides)

    def test_parameter_draw_precedes_all_vdfs_and_records_are_not_aliased(self):
        calls = []

        class Sampler(data.Fixed_ion_sampler):
            def sample_parameters(self):
                calls.append("parameters")
                return super().sample_parameters()

            def sample_electron_vdf(self):
                calls.append("e")
                self.assert_parameters = self.parameters["expected"]["e"]["temperature_ev"]
                return super().sample_electron_vdf()

            def sample_ion1_vdf(self):
                calls.append("i1")
                return super().sample_ion1_vdf()

            def sample_ion2_vdf(self):
                calls.append("i2")
                return super().sample_ion2_vdf()

        sampler = Sampler(np.random.default_rng(42), data._physical_config())
        first = sampler.sample()
        second = sampler.sample()
        self.assertEqual(calls, ["parameters", "e", "i1", "i2"] * 2)
        second["expected"]["e"]["temperature_ev"] = 1
        self.assertEqual(first["expected"]["e"]["temperature_ev"], 1370)
        actual = data.distribution_moments(*first["distributions"]["e"], data.ELECTRON_MASS)
        self.assertNotAlmostEqual(actual["actual_temperature_ev"], 1370)

    def test_anchor_reading_belongs_only_to_fig_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / "anchors.csv"
            csv.write_text("figure,shot_id,parameter,point_id,value\n"
                           "fig11,20170811020,T_e_ev,te,400\n"
                           "fig11,20170811020,T_i_ev,ti,100\n"
                           "fig6,20170811020,T_e_ev,ignored,9999\n", encoding="utf-8")
            config = data._physical_config_fig1_fig11()
            config["points_csv"] = str(csv)
            with mock.patch.object(data.Fig1_Fig11_sampler, "load_anchors", autospec=True,
                                   side_effect=data.Fig1_Fig11_sampler.load_anchors) as reader:
                sampler = data.Fig1_Fig11_sampler(np.random.default_rng(5), config)
                self.assertEqual(len(sampler.anchors["fig11"]["T_e_ev"]), 1)
                for _ in range(3):
                    sampler.sample()
                data.Fixed_ion_sampler(np.random.default_rng(5), data._physical_config()).sample()
                self.assertEqual(reader.call_count, 1)

    def test_hermitian_blocks_and_dynamic_loss_have_gradients(self):
        raw = torch.randn(3, 42, requires_grad=True)  # Three species, M=7.
        projected = model.hermitian_project(raw, fourier_modes=7)
        for index in range(3):
            part = raw[:, index * 14:(index + 1) * 14]
            torch.testing.assert_close(projected[:, index * 14:(index + 1) * 14],
                                       model.hermitian_project(part, fourier_modes=7))
        loss = train.coefficient_loss(raw, torch.zeros_like(raw), torch.zeros(42), torch.ones(42), .1, fourier_modes=7)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0)

    def test_checkpoint_checks_velocity_to_k_relation(self):
        payload = valid_checkpoint_payload()
        layout = data.dataset_layout(data._physical_config())
        payload["layout"] = layout
        payload["wavelengths_nm"] = torch.tensor(sum(layout["wavelength_bands_nm"], []), dtype=torch.float32)
        velocity = torch.tensor(layout["electron_dis"]["velocity_grid_m_s"], dtype=torch.float64)
        k, _ = data.ots_full_fourier.ft_x_n(velocity, torch.zeros_like(velocity), 80)
        payload["k_grid"] = k.float()
        model.validate_checkpoint(payload)
        bad = copy.deepcopy(payload)
        bad["k_grid"] *= 1.1
        with self.assertRaisesRegex(ValueError, "velocity axis"):
            model.validate_checkpoint(bad)
        bad = copy.deepcopy(payload)
        bad["layout"]["ion1_dis"]["name"] = "i2"
        with self.assertRaises(ValueError):
            model.validate_checkpoint(bad)

    def test_invalid_sampling_config_is_rejected_before_reading_anchors(self):
        for name, value in (("shape_probabilities", [.4, .4, .4]),
                            ("electron_temperature_factor", [-1, 2]),
                            ("density_fig11_cm3", [5e20, 3e20]),
                            ("ion_temperature_floor_ev", float("nan"))):
            config = data._physical_config_fig1_fig11()
            config["sampling"][name] = value
            with mock.patch.object(data.Fig1_Fig11_sampler, "load_anchors") as reader:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    data.Fig1_Fig11_sampler(np.random.default_rng(0), config)
                reader.assert_not_called()

    def test_incomplete_new_layout_does_not_fall_back_to_old_format(self):
        config = data._physical_config()
        del config["electron_dis"]["velocity_grid_m_s"]
        with self.assertRaisesRegex(ValueError, "Incomplete dataset layout"):
            data.load_dataset("unused.parquet", config)

    def test_generate_cli_defaults_and_config_passthrough(self):
        parser = main.build_parser()
        common = ["--run-id", "RUN", "--run-dir", "runs/output", "--dataset-id", "DATA"]
        args = parser.parse_args(["generate", *common])
        self.assertEqual(args.dataset_kind, "fixed_ion")
        self.assertIsNone(args.generation_config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            options = {"used_in_model": {"e": False}, "fourier_modes": 24}
            path.write_text(json.dumps(options), encoding="utf-8")
            args = parser.parse_args(["generate", *common, "--dataset-kind", "fig1_fig11", "--generation-config", str(path)])
            with mock.patch.object(main, "_preflight_output", return_value=Path(temporary)), \
                 mock.patch.object(main, "_publish_registry_entries"), \
                 mock.patch.object(main, "generate_dataset_pair", return_value={}) as generate:
                main.run_generate(args)
            self.assertEqual(generate.call_args.kwargs["dataset"], "fig1_fig11")
            self.assertEqual(generate.call_args.kwargs["generation_config"], options)

    def test_all_summary_links_each_species_artifact(self):
        args = main.build_parser().parse_args(["all", "--run-id", "RUN", "--run-dir", "runs/output",
            "--dataset-id", "DATA", "--checkpoint-id", "CHECKPOINT", "--dataset-kind", "fig1_fig11"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = {"train_path": root / "data/train.parquet", "test_path": root / "data/test.parquet",
                         "manifest_path": root / "data/dataset_manifest.json"}
            trained = {"checkpoint_path": root / "training/model.pt", "history_path": root / "training/history.json",
                       "learning_curves_path": root / "training/learning_curves.png",
                       "selected_candidate": {"best_validation_loss": 0.1}}
            evaluated = {"metrics_path": root / "evaluation/metrics.json"}
            for key, name in (("predictions", "predictions.npz"), ("comparison", "comparison.png"), ("sample_detail", "sample_detail.png")):
                evaluated[key + "_path"] = {sp: root / "evaluation" / sp / name for sp in ("e", "i1", "i2")}
            with mock.patch.object(main, "_preflight_output", return_value=root), \
                 mock.patch.object(main, "_publish_registry_entries"), \
                 mock.patch.object(main, "generate_dataset_pair", return_value=generated), \
                 mock.patch.object(main, "train_model", return_value=trained), \
                 mock.patch.object(main, "evaluate_checkpoint", return_value=evaluated):
                summary = main.run_all(args)
            self.assertEqual(summary["predictions"]["i2"], "evaluation/i2/predictions.npz")
            saved = json.loads((root / "pipeline_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["generation"]["dataset"], "fig1_fig11")


if __name__ == "__main__":
    unittest.main()
