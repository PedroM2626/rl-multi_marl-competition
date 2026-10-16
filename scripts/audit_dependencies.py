"""Query the OSV database for every pin in a requirements file and record what it says.

Why not `pip audit`: it resolves through PyPI's own advisory feed, which is a *subset* of OSV and lags it, so
a green `pip audit` is not evidence that no vulnerability is known — it is evidence that PyPI has not been
told. This script asks OSV, the upstream of both pip-audit and the GitHub advisory database, about the exact
`name==version` pairs the measured environment contains, so the answer is about the artefact that produced
the numbers in docs/ rather than about a fresh resolution of the dependency graph.

The environment is the pinned one (`requirements-measured.lock.txt`, `pip freeze` output), not
`requirements.txt`, because those two files describe different things — see [§ 9.1](../docs/09_reproducibility.md).

Nothing is installed, upgraded or removed here; the script is a read-only query and a record.

    python scripts/audit_dependencies.py                       # audit the measured lock
    python scripts/audit_dependencies.py --requirements requirements.txt --out results/other.json
    python scripts/audit_dependencies.py --print-only          # query, do not write the artefact

Requires network access to api.osv.dev. If the query fails the script exits non-zero and writes nothing: an
unverified audit is not a clean audit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{}"
ECOSYSTEM = "PyPI"
# OSV ids carry their own source; the ones that are also CVEs are worth naming in the report.
ID_PREFIXES = ("GHSA-", "PYSEC-", "CVE-")


def parse_pins(path: Path) -> list[tuple[str, str]]:
    """(name, version) for every `name==version` line, comments and options ignored.

    Names are lowercased because PyPI normalises them and OSV matches on the normalised form; a mixed-case
    pin like `ImageIO==2.37.0` otherwise returns an empty result that looks like "no known vulnerability".
    """
    pins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-") or "://" in line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9._-]+)\s*==\s*([A-Za-z0-9._*+!-]+)", line)
        if not match:
            raise SystemExit(f"{path.name}: cannot parse a pin from {line!r}")
        name, version = match.group(1).lower(), match.group(2)
        if "*" in version:
            raise SystemExit(f"{path.name}: {name}=={version} is not an exact pin, so OSV cannot answer "
                             f"about it; audit the freeze output, not a range")
        pins.append((name, version))
    if not pins:
        raise SystemExit(f"{path} contains no pins")
    return pins


def query_osv(pins: list[tuple[str, str]], timeout: int = 60) -> list[dict]:
    """One batched POST; the response is one result per query, in order."""
    body = json.dumps({"queries": [{"package": {"name": name, "ecosystem": ECOSYSTEM}, "version": version}
                                   for name, version in pins]}).encode("utf-8")
    request = urllib.request.Request(OSV_BATCH, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    if len(results) != len(pins):
        raise SystemExit(f"OSV returned {len(results)} results for {len(pins)} queries; refusing to "
                         f"align them by guesswork")
    return results


def severity_of(vuln: dict) -> str:
    """The highest ecosystem severity OSV records, or the GHSA/CVE alias level it carries."""
    levels: set[str] = set()
    for entry in vuln.get("severity", []):
        if isinstance(entry, dict) and entry.get("type") == "ECOSYSTEM":
            levels |= {score.get("severity", "").upper() for score in entry.get("score", [])}
    ordered = ["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE", "INFO"]
    for level in ordered:
        if level in levels:
            return level
    return str(vuln.get("database_specific", {}).get("severity") or "UNSPECIFIED")


def summarise(vuln_id: str, timeout: int = 60) -> dict:
    """Fetch the advisory itself for the fields a reader needs to decide whether it applies."""
    request = urllib.request.Request(OSV_VULN.format(vuln_id), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        vuln = json.loads(response.read().decode("utf-8"))
    fixed = sorted({version for affected in vuln.get("affected", [])
                    for range_ in affected.get("ranges", [])
                    for event in range_.get("events", [])
                    for version in [event.get("fixed")] if version})
    return {
        "id": vuln.get("id", vuln_id),
        "aliases": [alias for alias in vuln.get("aliases", []) if alias.startswith("CVE-")],
        "summary": (vuln.get("summary") or "")[:300],
        "severity": severity_of(vuln),
        "fixed_in": fixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements-measured.lock.txt",
                        help="the file whose exact pins should be asked about")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dependency_audit.json",
                        help="where the audit record goes")
    parser.add_argument("--print-only", action="store_true", help="query without writing the artefact")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = args.requirements if args.requirements.is_absolute() else ROOT / args.requirements
    pins = parse_pins(path)
    print(f"asking OSV about {len(pins)} pins from {path.name}", flush=True)
    try:
        results = query_osv(pins)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise SystemExit(f"OSV unreachable ({error}); no audit written — an unverified audit is not a "
                         f"clean one")

    packages: list[dict] = []
    advisories: list[str] = []
    for (name, version), result in zip(pins, results):
        ids = sorted({hit["id"] for hit in result.get("vulns", []) if hit.get("id")})
        entry = {"package": name, "version": version, "vulnerabilities": ids}
        if ids:
            entry["details"] = [summarise(vuln_id) for vuln_id in ids]
            for detail in entry["details"]:
                advisories.append(f"{name}=={version}: {detail['id']} ({detail['severity']})")
            print(f"  {name}=={version}: {', '.join(ids)}", flush=True)
        packages.append(entry)

    clean = sum(1 for entry in packages if not entry["vulnerabilities"])
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": path.name,
        "database": "OSV (api.osv.dev), queried per exact pin",
        "pins_queried": len(pins),
        "pins_with_no_known_vulnerability": clean,
        "vulnerable_pins": len(pins) - clean,
        "advisories": advisories,
        "packages": packages,
    }
    print(f"\n{clean}/{len(pins)} pins have no known OSV vulnerability; "
          f"{len(pins) - clean} have at least one")
    if args.print_only:
        print("--print-only: nothing written")
        return
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
