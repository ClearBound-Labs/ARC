"""Packaging contract: `import arc` is cheap, and every public name resolves."""
import subprocess
import sys

import arc


def test_import_arc_does_not_load_legacy_or_research_modules():
    code = (
        "import sys, arc; "
        "heavy = [m for m in ('scipy', 'arc.api.callback', 'arc.introspection', "
        "'arc.learning', 'arc.physics', 'arc.security.adversarial') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_every_public_name_resolves():
    missing = [name for name in arc.__all__ if getattr(arc, name, None) is None]
    assert missing == []


def test_unknown_attribute_raises_attribute_error():
    try:
        arc.definitely_not_a_name
    except AttributeError:
        return
    raise AssertionError("expected AttributeError")
