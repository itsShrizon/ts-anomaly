"""Warehouse module — persists detection events to a star-schema reporting DB.

This is an additive layer for MIS/FIS-style scheduled reporting. It does not
participate in training or inference; it only consumes outputs.
"""
