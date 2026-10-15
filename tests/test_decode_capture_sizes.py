import pytest


@pytest.mark.parametrize("capacity", [1, 2, 3, 5, 17, 64, 192])
def test_decode_capture_covers_every_request_count_without_exceeding_capacity(capacity):
    from nano_qwen.engine.cuda_graph import decode_capture_sizes

    sizes = decode_capture_sizes(capacity)
    for actual in range(1, capacity + 1):
        chosen = next(size for size in sizes if size >= actual)
        assert actual <= chosen <= capacity
    assert sizes[-1] == capacity
