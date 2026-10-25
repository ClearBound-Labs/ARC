"""AlertRouter: severity stacks (a WARNING channel also gets
CRITICAL alerts), and repeat alerts with the same dedup_key are
suppressed within the dedup window so a flapping condition doesn't spam."""
from arc.api.alerting import Alert, AlertRouter, Severity


def test_critical_alert_reaches_warning_and_critical_channels():
    warning_calls = []
    critical_calls = []
    router = AlertRouter(channels={
        Severity.WARNING: [lambda a: warning_calls.append(a)],
        Severity.CRITICAL: [lambda a: critical_calls.append(a)],
    })

    router.route(Alert(Severity.CRITICAL, "t", "m", dedup_key="k1"))

    assert len(warning_calls) == 1
    assert len(critical_calls) == 1


def test_info_alert_does_not_reach_warning_channel():
    warning_calls = []
    router = AlertRouter(channels={Severity.WARNING: [lambda a: warning_calls.append(a)]})

    router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1"))

    assert warning_calls == []


def test_dedup_suppresses_repeat_within_window():
    calls = []
    router = AlertRouter(
        channels={Severity.INFO: [lambda a: calls.append(a)]},
        dedup_window_seconds=100.0,
    )

    sent1 = router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1", timestamp=1000.0))
    sent2 = router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1", timestamp=1050.0))

    assert sent1 is True
    assert sent2 is False
    assert len(calls) == 1


def test_dedup_allows_after_window_elapses():
    calls = []
    router = AlertRouter(
        channels={Severity.INFO: [lambda a: calls.append(a)]},
        dedup_window_seconds=100.0,
    )

    router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1", timestamp=1000.0))
    sent2 = router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1", timestamp=1200.0))

    assert sent2 is True
    assert len(calls) == 2


if __name__ == "__main__":
    test_critical_alert_reaches_warning_and_critical_channels()
    test_info_alert_does_not_reach_warning_channel()
    test_dedup_suppresses_repeat_within_window()
    test_dedup_allows_after_window_elapses()
    print("ok")
