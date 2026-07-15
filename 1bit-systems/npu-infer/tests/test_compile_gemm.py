import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.compile_gemm import generate_xclbin, VALID_ROWS

def test_validate_dims():
    try:
        generate_xclbin(M=100, K=1024, N=2048)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "divisible" in str(e)

def test_validate_dims_k_not_divisible():
    try:
        generate_xclbin(M=256, K=100, N=2048)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "divisible" in str(e)

def test_validate_dims_n_not_divisible():
    try:
        generate_xclbin(M=256, K=1024, N=100)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "divisible" in str(e)

def test_validate_rows_invalid():
    try:
        generate_xclbin(M=256, K=1024, N=2048, rows=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rows must be in" in str(e)

def test_validate_group_count_too_small():
    try:
        generate_xclbin(M=128, K=1024, N=1024, rows=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "need at least 1 group" in str(e)

def test_validate_group_count_1row_ok():
    # (128//128//1)*(1024//128//8)=1*1=1 group — should pass validation
    xclbin, insts = generate_xclbin(M=128, K=1024, N=1024, rows=1)
    assert xclbin is not None

def test_valid_rows():
    assert 1 in VALID_ROWS
    assert 2 in VALID_ROWS
    assert "n1_core_bf16" in VALID_ROWS[1]

def test_valid_rows_2_maps():
    assert "n2_core_placed" in VALID_ROWS[2]
