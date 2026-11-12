"""Run the Spring gateway against the real FastAPI service, then stop the service and watch the fallback.

  python scripts/gateway_integration.py            # needs artifacts/ from export_model.py and make_history.py
                                                   # and gateway/target built by `mvn -f gateway/pom.xml test`

The gateway starts from target/classes with the classpath recorded in the surefire report. With the
Spring Boot repackage plugin available you can run the jar instead: set GATEWAY_CMD='java -jar gateway/target/tlc-gateway-0.1.0.jar'.
Phases: (1) model up, gateway answers must equal direct model answers; (2) model killed, answers must come
from fallback.csv with source seasonal_naive_fallback; (3) model restarted, the breaker must close again.
Writes results/gateway_integration.json."""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODEL_PORT, GW_PORT = 8011, 8012
ZONES = [4, 13, 43, 79, 132, 161, 186, 230, 236, 237]


def start_model() -> subprocess.Popen:
    env = {**os.environ, "MODEL_PATH": str(ART / "model.onnx"), "HISTORY_PATH": str(ART / "history.npz")}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "tlc_demand.service:create_app",
                          "--port", str(MODEL_PORT), "--log-level", "warning"], env=env, cwd=ROOT)
    wait(f"http://127.0.0.1:{MODEL_PORT}/healthz")
    return p


def wait(url: str, secs: int = 90) -> None:
    end = time.time() + secs
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


def gateway_command() -> list[str]:
    if os.environ.get("GATEWAY_CMD"):
        base = shlex.split(os.environ["GATEWAY_CMD"])
    else:
        report = glob.glob(str(ROOT / "gateway/target/surefire-reports/TEST-*.xml"))[0]
        cp = re.search(r'name="(?:surefire.test.class.path|java.class.path)" value="([^"]+)"', open(report).read()).group(1)
        cp = ":".join(x for x in cp.split(":") if "test-classes" not in x)
        base = ["java", "-cp", cp, "dev.bharath.gateway.GatewayApplication"]
    return base + [f"--server.port={GW_PORT}", f"--gateway.model-url=http://127.0.0.1:{MODEL_PORT}",
                   f"--gateway.fallback-file={ART / 'fallback.csv'}"]


def ask(zone: int) -> dict:
    r = httpx.get(f"http://127.0.0.1:{GW_PORT}/v1/forecast/{zone}", timeout=5)
    return {"status": r.status_code, **(r.json() if r.status_code == 200 else {})}


def fallback_counts() -> dict:
    text = httpx.get(f"http://127.0.0.1:{GW_PORT}/actuator/prometheus", timeout=5).text
    return {m.group(1): float(m.group(2)) for m in re.finditer(r'gateway_fallback_total\{[^}]*reason="(\w+)"[^}]*\} (\S+)', text)}


def main() -> None:
    fb = {int(a): float(b) for a, b in (l.split(",") for l in (ART / "fallback.csv").read_text().splitlines()[1:])}
    model = start_model()
    gw = subprocess.Popen(gateway_command(), cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out: dict = {"zones": ZONES, "gateway_started_from": "classpath" if not os.environ.get("GATEWAY_CMD") else "GATEWAY_CMD"}
    try:
        wait(f"http://127.0.0.1:{GW_PORT}/actuator/health")
        direct = {z: httpx.post(f"http://127.0.0.1:{MODEL_PORT}/v1/forecast", json={"zone": z}, timeout=5).json()["trips"] for z in ZONES}
        up = [ask(z) for z in ZONES]
        out["phase1_model_up"] = {
            "sources": sorted({a["source"] for a in up}),
            "max_abs_diff_vs_direct_model_call": max(abs(a["trips"] - direct[z]) for a, z in zip(up, ZONES)),
        }
        model.send_signal(signal.SIGKILL)
        model.wait()
        down = [ask(ZONES[i % len(ZONES)]) for i in range(40)]
        out["phase2_model_killed"] = {
            "requests": len(down),
            "sources": {s: sum(1 for a in down if a.get("source") == s) for s in {a.get("source") for a in down}},
            "all_values_equal_fallback_csv": all(a["trips"] == fb[a["zone"]] for a in down if a.get("source") == "seasonal_naive_fallback"),
            "fallback_reasons": fallback_counts(),
        }
        model = start_model()
        time.sleep(11)  # breaker's open interval is 10 s
        back = [ask(ZONES[i % len(ZONES)]) for i in range(20)]
        out["phase3_model_restarted"] = {"requests": len(back),
                                         "sources": {s: sum(1 for a in back if a.get("source") == s) for s in {a.get("source") for a in back}}}
    finally:
        for p in (gw, model):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
    (ROOT / "results/gateway_integration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
