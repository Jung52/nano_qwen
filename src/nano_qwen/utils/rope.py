def rope_config(config) -> tuple[float, float, list[int]]:
    """Normalize the current and early Qwen3.5 RoPE config layouts."""
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict):
        rope = getattr(config, "rope_scaling", None)
    if not isinstance(rope, dict):
        rope = {}
    base = rope.get("rope_theta", getattr(config, "rope_theta", 10_000_000.0))
    partial = rope.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.25))
    section = rope.get("mrope_section", [11, 11, 10])
    return float(base), float(partial), list(section)
