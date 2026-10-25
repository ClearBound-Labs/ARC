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

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import time
import json
import numpy as np
import torch
import torch.nn as nn

from arc.config import FailureMode, Config
from arc.api.callback import Arc
from arc.learning.simulator import FailureSimulator, SimulatedTrajectory
from arc.learning.labeler import TrajectoryLabeler
from arc.evaluation.metrics import EvaluationMetrics, MetricsCalculator

@dataclass
class BenchmarkResult:
    name: str
    metrics: EvaluationMetrics
    runtime_seconds: float
    config: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict(),
            "runtime_seconds": self.runtime_seconds,
            "config": self.config,
            "details": self.details,
        }

    def passed(
        self,
        min_precision: float = 0.6,
        min_recall: float = 0.6,
        max_false_alarm: float = 0.2,
        max_overhead: float = 5.0,
    ) -> bool:
        m = self.metrics
        return (
            m.precision >= min_precision and
            m.recall >= min_recall and
            m.false_alarm_rate <= max_false_alarm and
            m.overhead_percent <= max_overhead
        )

    def summary(self) -> str:
        status = "✓ PASS" if self.passed() else "✗ FAIL"
        return (
            f"{self.name}: {status}\n"
            f"  P={self.metrics.precision:.3f} R={self.metrics.recall:.3f} "
            f"F1={self.metrics.f1:.3f}\n"
            f"  FAR={self.metrics.false_alarm_rate:.3f} "
            f"Overhead={self.metrics.overhead_percent:.2f}%\n"
            f"  Runtime: {self.runtime_seconds:.1f}s"
        )

class BenchmarkSuite:
    """Benchmarks the rule-based `TrajectoryLabeler` against simulated
    trajectories whose failure mode is known, plus the legacy `Arc` API's
    overhead.

    It does not measure `arc.Controller` on real training runs, and its
    numbers must not be quoted as ARC's detection accuracy.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        n_trajectories: int = 100,
        verbose: bool = True,
    ):
        self.config = config or Config()
        self.n_trajectories = n_trajectories
        self.verbose = verbose

        self.simulator = FailureSimulator(seed=42)
        self.labeler = TrajectoryLabeler()

    def run_all(self) -> List[BenchmarkResult]:
        results = []

        if self.verbose:
            print("=" * 50)
            print("Arc Benchmark Suite")
            print("=" * 50)

        results.append(self.benchmark_detection_accuracy())
        results.append(self.benchmark_early_warning())
        results.append(self.benchmark_false_alarms())
        results.append(self.benchmark_calibration())
        results.append(self.benchmark_overhead())

        for mode in FailureMode:
            results.append(self.benchmark_single_mode(mode))

        if self.verbose:
            print("\n" + "=" * 50)
            print("RESULTS SUMMARY")
            print("=" * 50)
            for r in results:
                print(r.summary())

        return results

    def benchmark_detection_accuracy(self) -> BenchmarkResult:
        if self.verbose:
            print("\n[1/5] Detection Accuracy Benchmark...")

        start_time = time.time()

        trajectories = self.simulator.generate_dataset(
            n_trajectories=self.n_trajectories,
            success_ratio=0.3,
        )

        calculator = MetricsCalculator(threshold=0.5)

        for traj in trajectories:
            for epoch, signals in enumerate(traj.signals):
                partial_traj = traj.signals[:epoch+1]
                label = self.labeler.label_trajectory(partial_traj)

                pred = {}
                for mode in FailureMode:
                    if label.mode == mode:
                        pred[mode.name] = label.confidence
                    else:
                        pred[mode.name] = (1 - label.confidence) / (len(FailureMode) - 1)

                gt_mode = traj.failure_mode.name if traj.is_failure else None
                fail_epoch = traj.failure_epoch if traj.is_failure else None

                calculator.record_prediction(pred, epoch, gt_mode, fail_epoch)

        metrics = calculator.compute_metrics()
        runtime = time.time() - start_time

        return BenchmarkResult(
            name="Labeler Detection Accuracy (synthetic)",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={"n_trajectories": len(trajectories)},
        )

    def benchmark_early_warning(self) -> BenchmarkResult:
        if self.verbose:
            print("\n[2/5] Early Warning Benchmark...")

        start_time = time.time()

        trajectories = []
        for mode in FailureMode:
            gen = getattr(self.simulator, f"generate_{mode.name.lower()}_trajectory", None)
            if gen:
                for _ in range(self.n_trajectories // len(FailureMode)):
                    trajectories.append(gen(severity=0.7))

        calculator = MetricsCalculator(threshold=0.5)

        for traj in trajectories:
            for epoch, signals in enumerate(traj.signals):
                partial_traj = traj.signals[:epoch+1]
                label = self.labeler.label_trajectory(partial_traj)

                pred = {}
                for mode in FailureMode:
                    pred[mode.name] = label.confidence if label.mode == mode else 0.1

                calculator.record_prediction(
                    pred, epoch,
                    traj.failure_mode.name,
                    traj.failure_epoch
                )

        metrics = calculator.compute_metrics(lead_times=[3, 5, 10, 15])
        runtime = time.time() - start_time

        return BenchmarkResult(
            name="Labeler Early Warning (synthetic)",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={
                "mean_lead_time": metrics.mean_detection_time,
                "lead_time_targets": [3, 5, 10, 15],
            },
        )

    def benchmark_false_alarms(self) -> BenchmarkResult:
        if self.verbose:
            print("\n[3/5] False Alarm Benchmark...")

        start_time = time.time()

        trajectories = [
            self.simulator.generate_successful_trajectory(n_epochs=50)
            for _ in range(self.n_trajectories)
        ]

        calculator = MetricsCalculator(threshold=0.5)

        for traj in trajectories:
            for epoch, signals in enumerate(traj.signals):
                partial_traj = traj.signals[:epoch+1]
                label = self.labeler.label_trajectory(partial_traj)

                pred = {}
                for mode in FailureMode:
                    pred[mode.name] = label.confidence if label.mode == mode else 0.1

                calculator.record_prediction(pred, epoch, None, None)

        metrics = calculator.compute_metrics()
        runtime = time.time() - start_time

        return BenchmarkResult(
            name="Labeler False Alarm Rate (synthetic)",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={
                "n_healthy_trajectories": len(trajectories),
                "total_epochs": sum(t.n_epochs for t in trajectories),
            },
        )

    def benchmark_calibration(self) -> BenchmarkResult:
        if self.verbose:
            print("\n[4/5] Calibration Benchmark...")

        start_time = time.time()

        trajectories = self.simulator.generate_dataset(
            n_trajectories=self.n_trajectories,
            success_ratio=0.3,
        )

        calculator = MetricsCalculator(threshold=0.5)

        for traj in trajectories:
            label = self.labeler.label_trajectory(traj.signals)

            pred = {}
            for mode in FailureMode:
                pred[mode.name] = label.confidence if label.mode == mode else 0.05

            gt_mode = traj.failure_mode.name if traj.is_failure else None
            calculator.record_prediction(pred, traj.n_epochs - 1, gt_mode)

        metrics = calculator.compute_metrics()
        runtime = time.time() - start_time

        return BenchmarkResult(
            name="Labeler Calibration (synthetic)",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={"ece": metrics.ece},
        )

    def benchmark_overhead(self) -> BenchmarkResult:
        if self.verbose:
            print("\n[5/5] Overhead Benchmark...")

        start_time = time.time()

        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))

        n_epochs = 20
        n_steps = 100

        baseline_start = time.time()
        for epoch in range(n_epochs):
            for step in range(n_steps):
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        baseline_time = time.time() - baseline_start

        prophet = Arc(config=self.config, verbose=False)
        prophet.attach(model, optimizer)

        prophet_start = time.time()
        for epoch in range(n_epochs):
            for step in range(n_steps):
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                prophet.on_batch_end(loss.item())
            prophet.on_epoch_end(epoch)
        prophet_time = time.time() - prophet_start

        prophet.detach()

        overhead_pct = (prophet_time - baseline_time) / baseline_time * 100

        metrics = EvaluationMetrics(overhead_percent=overhead_pct)
        runtime = time.time() - start_time

        return BenchmarkResult(
            name="Overhead",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={
                "baseline_time": baseline_time,
                "prophet_time": prophet_time,
                "overhead_percent": overhead_pct,
                "n_epochs": n_epochs,
                "n_steps_per_epoch": n_steps,
            },
        )

    def benchmark_single_mode(self, mode: FailureMode) -> BenchmarkResult:
        if self.verbose:
            print(f"\n[Mode] {mode.name} Benchmark...")

        start_time = time.time()

        gen_method = f"generate_{mode.name.lower()}_trajectory"
        gen = getattr(self.simulator, gen_method, None)

        if gen is None:
            return BenchmarkResult(
                name=f"{mode.name} Detection",
                metrics=EvaluationMetrics(),
                runtime_seconds=0,
                config=self.config.to_dict(),
                details={"error": "No generator for this mode"},
            )

        trajectories = [gen(severity=0.7) for _ in range(self.n_trajectories // 2)]
        trajectories += [self.simulator.generate_successful_trajectory()
                        for _ in range(self.n_trajectories // 2)]

        calculator = MetricsCalculator(threshold=0.5)

        for traj in trajectories:
            label = self.labeler.label_trajectory(traj.signals)

            pred = {}
            for m in FailureMode:
                pred[m.name] = label.confidence if label.mode == m else 0.05

            gt_mode = traj.failure_mode.name if traj.is_failure else None
            calculator.record_prediction(pred, traj.n_epochs - 1, gt_mode, traj.failure_epoch)

        metrics = calculator.compute_metrics()
        runtime = time.time() - start_time

        return BenchmarkResult(
            name=f"{mode.name} Detection",
            metrics=metrics,
            runtime_seconds=runtime,
            config=self.config.to_dict(),
            details={"mode": mode.name},
        )

    def save_results(self, results: List[BenchmarkResult], path: str) -> None:
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)