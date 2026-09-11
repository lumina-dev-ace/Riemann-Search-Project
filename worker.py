"""Dedicated Render Background Worker for the Riemann search.

The web service remains the dashboard/control plane. This worker owns the
long-running numerical computation so the search is not tied to HTTP traffic
or the web service's free-tier idle/spin-down behavior.
"""

from __future__ import annotations

import asyncio

from app import START_N, init, latest, load_checkpoint, search, setstate


POLL_SECONDS = 5


async def watch_control() -> None:
    """Watch the GitHub checkpoint for a Start/Stop command from the dashboard."""
    while search.running:
        try:
            cp = await load_checkpoint()
            if cp and cp.get("paused") is True:
                search.running = False
                search.stopped = True
                search.reason = "Stopped by the dashboard"
                setstate("phase", "stopped")
                return
        except Exception as exc:
            # A transient checkpoint failure must not kill a running numerical
            # search. The normal checkpoint code will record errors separately.
            search.error = f"Worker control check failed: {type(exc).__name__}: {exc}"
        await asyncio.sleep(POLL_SECONDS)


async def main() -> None:
    init()
    print("Riemann numerical worker starting", flush=True)

    while True:
        if latest("candidates") is not None:
            print("Candidate already recorded; worker will remain idle.", flush=True)
            await asyncio.sleep(POLL_SECONDS)
            continue

        try:
            cp = await load_checkpoint()
            if cp is None:
                await asyncio.sleep(POLL_SECONDS)
                continue

            if cp.get("paused") is not False:
                await asyncio.sleep(POLL_SECONDS)
                continue

            # The dashboard has requested a run. The Search object performs
            # the actual FLINT/Arb work and its own periodic checkpointing.
            saved_n = max(START_N, int(cp.get("next_n", START_N)))
            setstate("next_n", saved_n)
            setstate("current_n", saved_n)
            setstate("phase", "worker starting")
            search.running = True
            search.stopped = False
            search.reason = ""
            search.error = ""

            watcher = asyncio.create_task(watch_control())
            try:
                await search.run()
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

        except Exception as exc:
            search.running = False
            search.stopped = True
            search.error = f"Worker crashed: {type(exc).__name__}: {exc}"
            setstate("phase", "worker error")
            print(search.error, flush=True)
            await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
