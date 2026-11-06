"""nano_qwen: a lightweight inference framework for Qwen3.5, FlashInfer-backed."""

__version__ = "0.1.0"


def main() -> int:
    from .chat import main as chat_main
    return chat_main()
