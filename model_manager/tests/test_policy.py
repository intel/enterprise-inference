"""Tests for CPU policy manager and quantity parsing."""
import pytest
from model_manager.core.policy import parse_cpu_quantity

def test_parse_cpu_quantity_integers():
    assert parse_cpu_quantity(8) == 8
    assert parse_cpu_quantity("16") == 16
    assert parse_cpu_quantity("1") == 1

def test_parse_cpu_quantity_millicores():
    assert parse_cpu_quantity("8000m") == 8
    assert parse_cpu_quantity("16000m") == 16
    assert parse_cpu_quantity("500m") == 1
    assert parse_cpu_quantity("1500m") == 2
    assert parse_cpu_quantity("16001m") == 17

def test_parse_cpu_quantity_decimals():
    assert parse_cpu_quantity("4.5") == 5
    assert parse_cpu_quantity("0.5") == 1
    assert parse_cpu_quantity("16.0") == 16
    assert parse_cpu_quantity("7.1") == 8

def test_parse_cpu_quantity_invalid():
    with pytest.raises(SystemExit):
        parse_cpu_quantity("abc")
    with pytest.raises(SystemExit):
        parse_cpu_quantity("0")
    with pytest.raises(SystemExit):
        parse_cpu_quantity("-5")
