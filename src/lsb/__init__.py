"""llm-security-benchmark: measure how an LLM application holds up under attack.

The package is organised around four pieces that stay deliberately separate:

- ``lsb.core``      the probe model, deterministic graders and statistics
- ``lsb.targets``   adapters for the system under test (mock, Anthropic, ...)
- ``lsb.defenses``  mitigations you can switch on to measure their effect
- ``lsb.probes``    the attack families, loaded from YAML suites
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
