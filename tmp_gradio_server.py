from __future__ import annotations

import time

import gradio_app


def main() -> None:
    gradio_app._ensure_workspace_dirs()
    app = gradio_app.build_app()
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
    )
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
