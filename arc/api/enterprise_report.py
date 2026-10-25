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

"""Unified EnterpriseReport aggregator: combines the recovery
ledger, the ROI report, policy-engine decisions, and routed alerts into one
artifact -- a JSON blob plus a self-contained static HTML page (inline SVG,
no CDN, no JS framework, no server) you can open in a browser. This is the
"pane of glass" in its zero-hosting degenerate form; the real fleet
dashboard needs a hosted control plane to serve live, multi-run,
multi-tenant views.
"""

import html as html_lib
import json
from dataclasses import asdict
from typing import Any, Dict, Optional


class EnterpriseReport:
    def __init__(self, controller, policy_engine=None, alert_router=None):
        self.controller = controller
        self.policy_engine = policy_engine
        self.alert_router = alert_router

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ledger": json.loads(self.controller.export_recovery_ledger()),
            "roi_report": self.controller.reliability_roi_report(),
        }
        if self.policy_engine is not None:
            data["policy_decisions"] = json.loads(self.policy_engine.export_log())
        if self.alert_router is not None:
            data["alerts"] = [asdict(a) for a in self.alert_router.sent_log]
        return data

    def _gpu_hours_bar_chart_svg(self, events) -> str:
        rollbacks = [e for e in events if e.get("action") == "rollback"]
        if not rollbacks:
            return "<p>No rollback events to chart.</p>"
        values = [e.get("gpu_hours_saved") for e in rollbacks]
        if any(v is None for v in values):
            return ("<p>GPU-hours saved unknown: set "
                    "<code>ControllerConfig.baseline_checkpoint_interval</code>.</p>")

        width, height, bar_gap = 480, 160, 4
        bar_width = max(1, (width - bar_gap * len(values)) // len(values))
        max_val = max(values) or 1.0

        bars = []
        for i, v in enumerate(values):
            bar_height = int((v / max_val) * (height - 20))
            x = i * (bar_width + bar_gap)
            y = height - bar_height
            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                f'fill="#4c78a8"><title>{v:.4f} GPU-hours</title></rect>'
            )

        return (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(bars) + "</svg>"
        )

    def to_html(self) -> str:
        data = self.to_dict()
        events = data["ledger"].get("events", [])
        chart = self._gpu_hours_bar_chart_svg(events)
        roi_escaped = html_lib.escape(data["roi_report"])
        json_escaped = html_lib.escape(json.dumps(data, indent=2))

        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ARC Enterprise Report</title></head>
<body>
<h1>ARC Enterprise Report</h1>
<h2>GPU-hours saved per rollback</h2>
{chart}
<h2>Reliability ROI</h2>
<pre>{roi_escaped}</pre>
<h2>Raw data</h2>
<pre>{json_escaped}</pre>
</body></html>"""

    def save(self, json_path: Optional[str] = None, html_path: Optional[str] = None) -> None:
        if json_path:
            with open(json_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        if html_path:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.to_html())
