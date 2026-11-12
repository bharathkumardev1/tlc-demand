"""Closed-loop load generator for the forecast service.

Requests ask for random zones, so this is synthetic load: it describes this generator against one
host and says nothing about real traffic. Reports client-side latency (includes HTTP and JSON).

python scripts/loadtest.py --url http://localhost:8000 --requests 2000 --concurrency 4"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx


def summarize(latencies_ms: list[float], errors: int, wall_s: float) -> dict:
    lat = sorted(latencies_ms)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 2) if lat else float("nan")

    return {"requests_ok": len(lat), "errors": errors, "wall_seconds": round(wall_s, 2),
            "throughput_rps": round(len(lat) / wall_s, 2) if wall_s > 0 else 0.0,
            "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99), "max": round(lat[-1], 2) if lat else float("nan")}}


async def run(url: str, zones: list[int], concurrency: int) -> dict:
    lat: list[float] = []
    errors = 0
    queue: asyncio.Queue[int] = asyncio.Queue()
    for z in zones:
        queue.put_nowait(z)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while True:
            try:
                z = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            try:
                r = await client.post("/v1/forecast", json={"zone": z})
                if r.status_code == 404:  # zone with no history is a valid answer, not a failure
                    lat.append((time.perf_counter() - t0) * 1000)
                    continue
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)
            except (httpx.HTTPError, ValueError):
                errors += 1

    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        for z in zones[:50]:  # warm-up, not counted
            await client.post("/v1/forecast", json={"zone": z})
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return summarize(lat, errors, wall)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    zones = [rng.randint(1, 265) for _ in range(a.requests)]
    result = asyncio.run(run(a.url, zones, a.concurrency))
    print("SYNTHETIC LOAD (random zones, single host, not real traffic)")
    print(json.dumps({"url": a.url, "concurrency": a.concurrency, **result}, indent=2))


if __name__ == "__main__":
    main()
