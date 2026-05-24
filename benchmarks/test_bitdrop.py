# test_bitdrop.py

# tests/test_bitdrop.py

import pytest
from bitdrop import (
    Cell, Net, Row, Region,
    compute_bit_importance,
    bitdrop_collapse_placement,
    global_cost,
)

def build_small_design():
    cells = [Cell(f"C{i}") for i in range(8)]
    nets = [Net(f"N{i}") for i in range(4)]

    nets[0].cells = [cells[0], cells[1], cells[2]]
    nets[1].cells = [cells[2], cells[3]]
    nets[2].cells = [cells[4], cells[5], cells[6]]
    nets[3].cells = [cells[1], cells[7]]

    for n in nets:
        for c in n.cells:
            c.nets.append(n)

    nets[0].timing_slack = -0.12
    nets[1].timing_slack = 0.05
    nets[2].timing_slack = -0.30
    nets[3].timing_slack = -0.02

    nets[0].activity = 0.4
    nets[1].activity = 0.1
    nets[2].activity = 0.8
    nets[3].activity = 0.2

    rows_R0 = [Row(0, 10)]
    rows_R1 = [Row(1, 10)]
    regions = [
        Region("R0", rows_R0, power_strength=1.0),
        Region("R1", rows_R1, power_strength=2.0),
    ]

    return cells, nets, regions

def test_all_cells_placed():
    cells, nets, regions = build_small_design()
    cells, nets, regions, _ = bitdrop_collapse_placement(cells, nets, regions)
    assert all(c.row is not None for c in cells)

def test_no_overlaps():
    cells, nets, regions = build_small_design()
    cells, nets, regions, _ = bitdrop_collapse_placement(cells, nets, regions)
    for R in regions:
        for row in R.rows:
            seen = set()
            for c in row.cells:
                if c is None:
                    continue
                assert c not in seen
                seen.add(c)

def test_bit_importance_positive_for_negative_slack():
    cells, nets, regions = build_small_design()
    compute_bit_importance(nets)
    for n in nets:
        if n.timing_slack < 0:
            assert n.bit_importance > 0

def test_cost_history_monotonic():
    cells, nets, regions = build_small_design()
    cells, nets, regions, history = bitdrop_collapse_placement(
        cells, nets, regions, record_history=True
    )
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-6
