from __future__ import annotations

import gradio_app


def main() -> None:
    gradio_app._ensure_workspace_dirs()
    app = gradio_app.build_app()
    _, local_url, share_url = app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True,
        prevent_thread_lock=True,
    )
    print(f"* Running on local URL:  {local_url}")
    print(
        "* Bound on all interfaces via 0.0.0.0:7860. "
        "Open localhost or this machine's LAN IP in a browser, not the bind address."
    )
    if share_url:
        print(f"* Running on public URL: {share_url}")
    else:
        print("* To create a public link, set `share=True` in `launch()`.")
    app.block_thread()


if __name__ == "__main__":
    main()
