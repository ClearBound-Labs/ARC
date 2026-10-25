# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

"""
ARC Test Suite

Comprehensive tests for all ARC components.
Run with: pytest tests/test_arc.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )


@pytest.fixture
def simple_cnn():
    """Simple CNN for testing."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 16 * 16, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return SimpleCNN()


# =============================================================================
# WeightRollback Tests
# =============================================================================

class TestWeightRollback:
    """Tests for WeightRollback intervention system."""
    
    def test_import(self):
        """Test that WeightRollback can be imported."""
        from arc.intervention import WeightRollback
        assert WeightRollback is not None
    
    def test_initialization(self, simple_model):
        """Test WeightRollback initialization."""
        from arc.intervention import WeightRollback
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer)
        
        assert rollback is not None
        assert len(rollback.state.checkpoints) == 1  # Initial checkpoint
    
    def test_checkpoint_saving(self, simple_model):
        """Test that checkpoints are saved periodically."""
        from arc.intervention import WeightRollback, RollbackConfig
        
        config = RollbackConfig(checkpoint_frequency=5)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)
        
        # Run 10 steps
        for _ in range(10):
            x = torch.randn(4, 100)
            loss = simple_model(x).mean()
            loss.backward()
            rollback.step(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        # Should have initial + 2 periodic checkpoints
        assert len(rollback.state.checkpoints) >= 2
    
    def test_failure_detection_nan(self, simple_model):
        """Test NaN loss detection."""
        from arc.intervention import WeightRollback
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, verbose=False)
        
        # Normal step
        action1 = rollback.step(torch.tensor(1.0))
        assert not action1.rolled_back
        
        # NaN step
        action2 = rollback.step(torch.tensor(float('nan')))
        assert action2.failure_detected
        assert action2.failure_type == "nan_loss"
    
    def test_rollback_restores_weights(self, simple_model):
        """Test that rollback actually restores weights."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(checkpoint_frequency=1, loss_explosion_threshold=10.0)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)

        # Save initial weights before any steps
        initial_weights = simple_model[0].weight.data.clone()

        # Step 1: save a checkpoint with CLEAN weights
        rollback.step(torch.tensor(1.0))

        # Step 2: corrupt weights AFTER the checkpoint is saved
        with torch.no_grad():
            simple_model[0].weight.data.fill_(999.0)

        # Step 3: trigger rollback with an explosive loss
        action = rollback.step(torch.tensor(100.0))

        # Step 4: rollback MUST have fired — assert unconditionally
        assert action.rolled_back, "Expected rollback to trigger but it did not"

        # Step 5: weights must no longer be the corrupted value
        assert not torch.allclose(
            simple_model[0].weight.data,
            torch.full_like(simple_model[0].weight.data, 999.0)
        ), "Weights were not restored — still showing corrupted 999.0 values"

        # Step 6: weights must be back to the clean checkpoint values
        assert torch.allclose(
            simple_model[0].weight.data, initial_weights, atol=1e-5
        ), "Weights were not restored to the pre-corruption checkpoint values"


# =============================================================================
# GradientForecaster Tests
# =============================================================================

class TestGradientForecaster:
    """Tests for GradientForecaster prediction system."""
    
    def test_import(self):
        """Test that GradientForecaster can be imported."""
        from arc.prediction import GradientForecaster
        assert GradientForecaster is not None
    
    def test_initialization(self, simple_model):
        """Test GradientForecaster initialization."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        assert forecaster is not None
        assert forecaster.step_count == 0
    
    def test_update(self, simple_model):
        """Test forecaster update."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        
        x = torch.randn(4, 100)
        loss = simple_model(x).mean()
        loss.backward()
        
        update_info = forecaster.update()
        assert "grad_norm" in update_info
        assert update_info["step"] == 1
    
    def test_prediction(self, simple_model):
        """Test gradient prediction."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        
        # Run a few steps to build history
        for _ in range(10):
            x = torch.randn(4, 100)
            loss = simple_model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            forecaster.update()
            optimizer.step()
        
        forecast = forecaster.predict()
        assert hasattr(forecast, 'will_explode')
        assert hasattr(forecast, 'steps_until')
        assert hasattr(forecast, 'confidence')


# =============================================================================
# LiteArc Tests
# =============================================================================

class TestLiteArc:
    """Tests for LiteArc low-overhead monitoring."""
    
    def test_import(self):
        """Test that LiteArc can be imported."""
        from arc.api.lite import LiteArc
        assert LiteArc is not None
    
    def test_step_skipping(self, simple_model):
        """Test that LiteArc skips non-check steps."""
        from arc.api.lite import LiteArc, LiteConfig
        
        config = LiteConfig(check_every_n_steps=10)
        arc = LiteArc(simple_model, config=config)
        
        # First 9 steps should not be checked
        for i in range(9):
            result = arc.step(torch.tensor(1.0))
            assert result["checked"] == False
        
        # 10th step should be checked
        result = arc.step(torch.tensor(1.0))
        assert result["checked"] == True
    
    def test_nan_detection(self, simple_model):
        """Test NaN loss detection in LiteArc."""
        from arc.api.lite import LiteArc, LiteConfig
        
        config = LiteConfig(check_every_n_steps=1)  # Check every step
        arc = LiteArc(simple_model, config=config)
        
        result = arc.step(torch.tensor(float('nan')))
        assert result["alert"] == "nan_loss"


# =============================================================================
# ArcV2 Integration Tests
# =============================================================================

class TestArcV2:
    """Tests for main ArcV2 API."""
    
    def test_import(self):
        """Test ArcV2 import."""
        from arc import ArcV2
        assert ArcV2 is not None
    
    def test_auto_creation(self, simple_model):
        """Test ArcV2.auto() factory."""
        from arc import ArcV2
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2.auto(simple_model, optimizer)
        
        assert arc is not None
    
    def test_step(self, simple_model):
        """Test ArcV2 step."""
        from arc import ArcV2
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2.auto(simple_model, optimizer)
        
        result = arc.step(torch.tensor(1.0))
        assert "step" in result


# =============================================================================
# ArcV2.auto() Large-Model Regression Tests  (Issue #1)
# =============================================================================

class TestArcV2AutoLargeModel:
    """
    Regression tests for issue #1:
    "[BUG] ArcV2.auto() passes raw dicts to Config instead of config dataclass instances"

    Verifies that both large-model branches in ArcV2.auto() correctly construct
    SignalConfig and FeatureConfig dataclass instances rather than raw dicts, and
    that the field name 'activation_sample_ratio' (not 'activation_sample_rate')
    is used.
    """

    @staticmethod
    def _make_model_gt10m() -> nn.Module:
        """
        Build a model with more than 10M parameters.
        Linear(1024, 3000)  => 1024 * 3000 + 3000 = 3,075,000
        Linear(3000, 3000)  => 3000 * 3000 + 3000 = 9,003,000
        Linear(3000, 10)    =>    3000 * 10 + 10  =    30,010
        Total               => ~12,108,010  > 10M
        """
        return nn.Sequential(
            nn.Linear(1024, 3000),
            nn.ReLU(),
            nn.Linear(3000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 10),
        )

    def test_large_model_param_count(self):
        """Sanity check: confirm helper model exceeds 10M parameters."""
        model = self._make_model_gt10m()
        n = sum(p.numel() for p in model.parameters())
        assert n > 10_000_000, f"Expected >10M params, got {n}"

    def test_auto_no_crash_gt10m(self):
        """
        ArcV2.auto() must not crash for models with >10M parameters.

        Before the fix, this raised:
            AttributeError: 'dict' object has no attribute 'compute_curvature_proxy'
        inside Arc._init_collectors() because config.signal was a raw dict.
        """
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)   # must not raise
        assert arc is not None

    def test_signal_is_SignalConfig_instance_gt10m(self):
        """config.signal must be a SignalConfig instance, not a raw dict."""
        from arc import ArcV2
        from arc.config import SignalConfig

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert isinstance(arc.config.signal, SignalConfig), (
            f"Expected SignalConfig, got {type(arc.config.signal).__name__}"
        )

    def test_feature_is_FeatureConfig_instance_gt10m(self):
        """config.feature must be a FeatureConfig instance, not a raw dict."""
        from arc import ArcV2
        from arc.config import FeatureConfig

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert isinstance(arc.config.feature, FeatureConfig), (
            f"Expected FeatureConfig, got {type(arc.config.feature).__name__}"
        )

    def test_activation_sample_ratio_is_valid_float_gt10m(self):
        """
        config.signal.activation_sample_ratio must be a valid float in (0, 1].

        Before the fix, 'activation_sample_rate' (wrong name) was stored in a
        raw dict — accessing .activation_sample_ratio would have raised
        AttributeError or returned the dataclass default instead of 0.1.
        """
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        ratio = arc.config.signal.activation_sample_ratio
        assert isinstance(ratio, float), (
            f"Expected float, got {type(ratio).__name__}"
        )
        assert 0.0 < ratio <= 1.0, (
            f"activation_sample_ratio out of range: {ratio}"
        )

    def test_activation_sample_ratio_value_gt10m(self):
        """config.signal.activation_sample_ratio must equal 0.1 for >10M models."""
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert arc.config.signal.activation_sample_ratio == 0.1

    def test_window_size_gt10m(self):
        """config.feature.window_size must equal 10 for >10M-param models."""
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert arc.config.feature.window_size == 10


# =============================================================================
# Config Numeric Validation Tests  (Issue #2)
# =============================================================================

class TestConfigValidation:
    """
    Tests for __post_init__ numeric validation added to config dataclasses.

    Covers:
      - SignalConfig: activation_sample_ratio, collect_every_n_steps
      - FeatureConfig: window_size, min_history
      - PredictionConfig: dropout_rate, confidence_level, mc_dropout_samples
      - FailureThresholds: loss_explosion_factor, numeric threshold fields
      - OverheadConfig: max_overhead_percent, overhead_check_interval
      - Config.from_dict() propagates ValueError for invalid nested values
      - Default Config(), Config.low_overhead(), Config.high_accuracy() still pass
    """

    # ------------------------------------------------------------------
    # SignalConfig
    # ------------------------------------------------------------------

    def test_signal_ratio_zero_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=0.0)

    def test_signal_ratio_negative_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=-0.1)

    def test_signal_ratio_above_one_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=1.5)

    def test_signal_ratio_exactly_one_valid(self):
        """Boundary: ratio == 1.0 must be accepted (spec is (0, 1])."""
        from arc.config import SignalConfig
        cfg = SignalConfig(activation_sample_ratio=1.0)
        assert cfg.activation_sample_ratio == 1.0

    def test_signal_collect_steps_zero_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="collect_every_n_steps"):
            SignalConfig(collect_every_n_steps=0)

    def test_signal_collect_steps_negative_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="collect_every_n_steps"):
            SignalConfig(collect_every_n_steps=-1)

    # ------------------------------------------------------------------
    # FeatureConfig
    # ------------------------------------------------------------------

    def test_feature_window_size_zero_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="window_size"):
            FeatureConfig(window_size=0)

    def test_feature_window_size_negative_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="window_size"):
            FeatureConfig(window_size=-1)

    def test_feature_min_history_zero_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="min_history"):
            FeatureConfig(min_history=0)

    def test_feature_min_history_negative_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="min_history"):
            FeatureConfig(min_history=-5)

    # ------------------------------------------------------------------
    # PredictionConfig
    # ------------------------------------------------------------------

    def test_prediction_dropout_negative_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="dropout_rate"):
            PredictionConfig(dropout_rate=-0.1)

    def test_prediction_dropout_above_one_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="dropout_rate"):
            PredictionConfig(dropout_rate=1.5)

    def test_prediction_dropout_zero_valid(self):
        """Boundary: dropout_rate == 0.0 must be accepted (spec is [0, 1])."""
        from arc.config import PredictionConfig
        cfg = PredictionConfig(dropout_rate=0.0)
        assert cfg.dropout_rate == 0.0

    def test_prediction_confidence_zero_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="confidence_level"):
            PredictionConfig(confidence_level=0.0)

    def test_prediction_confidence_above_one_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="confidence_level"):
            PredictionConfig(confidence_level=1.1)

    def test_prediction_confidence_exactly_one_valid(self):
        """Boundary: confidence_level == 1.0 must be accepted (spec is (0, 1])."""
        from arc.config import PredictionConfig
        cfg = PredictionConfig(confidence_level=1.0)
        assert cfg.confidence_level == 1.0

    def test_prediction_mc_samples_zero_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="mc_dropout_samples"):
            PredictionConfig(mc_dropout_samples=0)

    def test_prediction_mc_samples_negative_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="mc_dropout_samples"):
            PredictionConfig(mc_dropout_samples=-1)

    # ------------------------------------------------------------------
    # FailureThresholds
    # ------------------------------------------------------------------

    def test_thresholds_loss_factor_zero_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="loss_explosion_factor"):
            FailureThresholds(loss_explosion_factor=0.0)

    def test_thresholds_loss_factor_negative_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="loss_explosion_factor"):
            FailureThresholds(loss_explosion_factor=-1.0)

    def test_thresholds_vanishing_grad_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="vanishing_grad_threshold"):
            FailureThresholds(vanishing_grad_threshold=0.0)

    def test_thresholds_exploding_grad_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="exploding_grad_threshold"):
            FailureThresholds(exploding_grad_threshold=-1.0)

    def test_thresholds_activation_similarity_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="activation_similarity_threshold"):
            FailureThresholds(activation_similarity_threshold=0.0)

    def test_thresholds_effective_rank_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="effective_rank_collapse_ratio"):
            FailureThresholds(effective_rank_collapse_ratio=-0.1)

    def test_thresholds_overfit_gap_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="overfit_gap_threshold"):
            FailureThresholds(overfit_gap_threshold=0.0)

    # ------------------------------------------------------------------
    # OverheadConfig
    # ------------------------------------------------------------------

    def test_overhead_percent_negative_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="max_overhead_percent"):
            OverheadConfig(max_overhead_percent=-1.0)

    def test_overhead_percent_zero_valid(self):
        """Boundary: max_overhead_percent == 0.0 must be accepted (spec is >= 0)."""
        from arc.config import OverheadConfig
        cfg = OverheadConfig(max_overhead_percent=0.0)
        assert cfg.max_overhead_percent == 0.0

    def test_overhead_interval_zero_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="overhead_check_interval"):
            OverheadConfig(overhead_check_interval=0)

    def test_overhead_interval_negative_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="overhead_check_interval"):
            OverheadConfig(overhead_check_interval=-10)

    # ------------------------------------------------------------------
    # Config.from_dict() propagation
    # ------------------------------------------------------------------

    def test_from_dict_invalid_signal_ratio_raises(self):
        """Config.from_dict() must propagate ValueError for invalid nested values."""
        from arc.config import Config
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            Config.from_dict({"signal": {"activation_sample_ratio": -1.0}})

    def test_from_dict_invalid_dropout_raises(self):
        from arc.config import Config
        with pytest.raises(ValueError, match="dropout_rate"):
            Config.from_dict({"prediction": {"dropout_rate": 2.0}})

    # ------------------------------------------------------------------
    # Preset smoke tests — must NOT raise
    # ------------------------------------------------------------------

    def test_default_config_passes(self):
        """Config() with all defaults must construct without error."""
        from arc.config import Config
        cfg = Config()
        assert cfg is not None

    def test_low_overhead_preset_passes(self):
        """Config.low_overhead() must construct without error."""
        from arc.config import Config
        cfg = Config.low_overhead()
        assert cfg is not None

    def test_high_accuracy_preset_passes(self):
        """Config.high_accuracy() must construct without error."""
        from arc.config import Config
        cfg = Config.high_accuracy()
        assert cfg is not None


# =============================================================================
# Conformal Predictor Tests
# =============================================================================

class TestConformalPredictors:
    """Tests for ConformalPredictor, ConformalTTFPredictor, and ConformallyCalibratedPredictor."""
    
    def test_conformal_ttf_small_sample_asymmetric_no_crash(self):
        """Test that ConformalTTFPredictor doesn't crash on small calibration size for asymmetric interval."""
        from arc.prediction.conformal import ConformalTTFPredictor
        import numpy as np

        # Under target coverage = 0.9 (alpha = 0.1), symmetric = False,
        # q_alpha_upper is 0.95. For n = 11, 0.95 * (1 + 1/11) = 1.036 > 1.0,
        # which would crash np.quantile if not clamped.
        predictor = ConformalTTFPredictor(target_coverage=0.9, symmetric=False)
        predictions = np.zeros(11)
        actuals = np.random.randn(11)

        # This should run successfully without raising ValueError: Quantiles must be in the range [0, 1]
        q_lower, q_upper = predictor.calibrate(predictions, actuals)
        assert q_lower <= q_upper

    def test_conformal_ttf_small_sample_symmetric_no_crash(self):
        """Test that ConformalTTFPredictor doesn't crash on small calibration size for symmetric interval."""
        from arc.prediction.conformal import ConformalTTFPredictor
        import numpy as np

        # Under target coverage = 0.9 (alpha = 0.1), symmetric = True,
        # (1 - alpha) * (1 + 1/5) = 0.9 * 1.2 = 1.08 > 1.0, which would crash.
        predictor = ConformalTTFPredictor(target_coverage=0.9, symmetric=True)
        predictions = np.zeros(5)
        actuals = np.random.randn(5)

        q_lower, q_upper = predictor.calibrate(predictions, actuals)
        assert q_lower <= q_upper

    def test_conformally_calibrated_predictor_integration(self):
        """Test calibration process through ConformallyCalibratedPredictor with small sample sizes."""
        from arc.prediction.conformal import ConformallyCalibratedPredictor
        import numpy as np

        predictor = ConformallyCalibratedPredictor(n_failure_modes=2)
        
        class_probs = np.random.dirichlet(np.ones(2), size=15)
        class_labels = np.random.randint(0, 2, size=15)
        
        # Mode 0 has 12 predictions (which triggers calibration since 12 > 10,
        # but is less than 19, so it would crash asymmetric calibration without clamping)
        ttf_predictions = {
            0: np.random.randn(12),
            1: np.random.randn(5)  # Mode 1 has 5 predictions (does not calibrate since <= 10)
        }
        ttf_actuals = {
            0: np.random.randn(12),
            1: np.random.randn(5)
        }

        # Calibrate should not crash
        predictor.calibrate(class_probs, class_labels, ttf_predictions, ttf_actuals)
        assert predictor._is_calibrated

        # Predict should run successfully
        res = predictor.predict(class_probs[0], [0.5, 0.5])
        assert "ttf_intervals" in res


# =============================================================================
# Issue #6 Regression Tests — Persistence & Configuration
# =============================================================================

class TestRollbackMaxCheckpoints:
    """
    Regression tests for Issue #6 Bug 1:
    RollbackConfig.max_checkpoints was silently ignored because the deque
    in RollbackState hardcoded maxlen=3.
    """

    def test_default_max_checkpoints(self, simple_model):
        """Default config should use max_checkpoints=3."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())
        rb = WeightRollback(simple_model, optimizer, config, verbose=False)

        assert rb.state.checkpoints.maxlen == 3

    def test_custom_max_checkpoints_honoured(self, simple_model):
        """Setting max_checkpoints=5 must create a deque with maxlen=5."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(max_checkpoints=5)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rb = WeightRollback(simple_model, optimizer, config, verbose=False)

        assert rb.state.checkpoints.maxlen == 5

    def test_custom_max_checkpoints_retains_more(self, simple_model):
        """With max_checkpoints=10, we should retain more than 3 checkpoints."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(max_checkpoints=10, checkpoint_frequency=1)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rb = WeightRollback(simple_model, optimizer, config, verbose=False)

        # Run enough steps to save several checkpoints
        for _ in range(20):
            x = torch.randn(4, 100)
            loss = simple_model(x).mean()
            loss.backward()
            rb.step(loss)
            optimizer.step()
            optimizer.zero_grad()

        # With maxlen=10 we should have more than the old hardcoded 3
        assert len(rb.state.checkpoints) > 3


class TestArcV2EWCSaveLoad:
    """
    Regression tests for Issue #6 Bug 2:
    ArcV2.save() computed ewc_state but never wrote it to disk.
    No load() method existed.
    """

    def test_save_creates_ewc_state_file(self, simple_model, tmp_path):
        """save() must write ewc_state.pt when EWC is active."""
        from arc import ArcV2
        import os

        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2(
            continual_learning=True,
            verbose=False,
        )
        arc.attach(simple_model, optimizer)

        # Create a dummy dataloader for consolidation
        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 100), torch.randint(0, 10, (20,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        arc.begin_task("task_1")
        arc.consolidate_task(loader)
        arc.save(str(tmp_path / "arc_ckpt"))

        assert os.path.exists(tmp_path / "arc_ckpt" / "ewc_state.pt")
        assert os.path.exists(tmp_path / "arc_ckpt" / "ewc_meta.json")

    def test_load_restores_ewc_n_tasks(self, simple_model, tmp_path):
        """load() must restore EWC task count from saved state."""
        from arc import ArcV2

        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2(
            continual_learning=True,
            verbose=False,
        )
        arc.attach(simple_model, optimizer)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 100), torch.randint(0, 10, (20,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        arc.begin_task("task_1")
        arc.consolidate_task(loader)
        assert arc._ewc.n_tasks == 1

        save_path = str(tmp_path / "arc_ckpt")
        arc.save(save_path)

        # Create a fresh ArcV2 and restore
        model2 = nn.Sequential(
            nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10),
        )
        optimizer2 = torch.optim.Adam(model2.parameters())
        arc2 = ArcV2(
            continual_learning=True,
            verbose=False,
        )
        arc2.attach(model2, optimizer2)
        arc2.load(save_path, model=model2)

        assert arc2._ewc is not None
        assert arc2._ewc.n_tasks == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])