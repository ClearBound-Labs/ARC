"""RL stability detectors: KL blowup, entropy death, reward
hacking, and advantage explosion each fire on their characteristic
synthetic signal and stay quiet on a healthy one."""
from arc.signals.rl_stability import RLStabilityMonitor


def test_healthy_signals_report_no_issues():
    monitor = RLStabilityMonitor()
    report = None
    for i in range(20):
        report = monitor.update(kl=0.02, entropy=1.5, reward=0.5, true_objective=0.5, advantage=0.1)
    assert report.is_healthy


def test_kl_blowup_detected():
    monitor = RLStabilityMonitor(kl_threshold=0.1)
    report = monitor.update(kl=0.5)
    assert report.kl_blowup is True


def test_entropy_collapse_detected_by_floor():
    monitor = RLStabilityMonitor(entropy_floor=0.05)
    report = monitor.update(entropy=0.001)
    assert report.entropy_collapse is True


def test_entropy_collapse_detected_by_slope():
    monitor = RLStabilityMonitor(entropy_floor=0.0, entropy_slope_threshold=-0.01)
    report = None
    for i in range(20):
        report = monitor.update(entropy=2.0 - i * 0.1)  # steadily decaying
    assert report.entropy_collapse is True


def test_reward_hacking_detected():
    monitor = RLStabilityMonitor()
    report = None
    for i in range(20):
        report = monitor.update(reward=i * 0.1, true_objective=-i * 0.1)
    assert report.reward_hacking is True


def test_advantage_explosion_detected():
    monitor = RLStabilityMonitor(advantage_zscore_threshold=3.0)
    report = None
    for i in range(10):
        report = monitor.update(advantage=0.1 + (i % 2) * 0.01)
    report = monitor.update(advantage=50.0)  # sudden spike
    assert report.advantage_explosion is True


if __name__ == "__main__":
    test_healthy_signals_report_no_issues()
    test_kl_blowup_detected()
    test_entropy_collapse_detected_by_floor()
    test_entropy_collapse_detected_by_slope()
    test_reward_hacking_detected()
    test_advantage_explosion_detected()
    print("ok")
