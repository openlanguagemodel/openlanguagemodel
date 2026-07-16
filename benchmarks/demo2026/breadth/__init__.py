"""Breadth validation for OpenLanguageModel.

Validates the documented 9 model families and 27 named presets by:
  - Checking constructor kwargs against the manifest (no large allocation)
  - Running forward/backward on tiny reduced models
  - Verifying tied embeddings
  - Checking checkpoint round-trip bitwise identity
  - Validating formula-based parameter counts against actual models
"""
