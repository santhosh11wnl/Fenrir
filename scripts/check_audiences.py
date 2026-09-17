#!/usr/bin/env python
"""Prove, against a project's real index, that role-scoped retrieval holds.

The unit tests cover the logic; this covers the *wiring* -- that the documents
were actually tagged during ingest, and that a restricted role genuinely cannot
pull a privileged chunk out of the store it is pointed at. Those are the parts
that break silently: a mistagged `ingest.yaml` source produces a corpus that
looks fine until someone reads material they should not.

Run it inside the container, so it queries the same store the API queries:

    docker compose exec -T api python scripts/check_audiences.py shop

An empty index makes every check vacuously pass, so that is a hard failure
rather than a row of reassuring ticks.

Exit status is 0 only if every expectation held, so this can gate a deploy.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from chatbot_core.config import ProjectConfig
from chatbot_core.retrieval import build_embedder, build_store

#: Probes phrased the way each audience's material is actually written, so a
#: miss means "not retrievable", not "badly worded query".
PROBES: dict[str, str] = {
    "public": "how long does standard delivery take",
    "staff": "goodwill credit limit without approval escalation",
    "internal": "gross margin supplier payment terms",
}


async def main(project: str, quiet: bool) -> int:
    config = ProjectConfig.load(f"projects/{project}/config.yaml")
    store = build_store(
        config.retrieval, config.collection, build_embedder(config.retrieval.embedding)
    )
    await store.ensure_ready()

    total = await store.count()
    print(f"project {project!r}: {total} chunks indexed")
    if total == 0:
        print("\nFAIL: the index is empty, so every check below would pass vacuously.")
        print("Ingest the corpus first:  docker compose run --rm ingest")
        return 1

    # What the corpus actually claims, independent of any role. Catches the
    # mistagged-source case before it is mistaken for a filtering result.
    #
    # Swept across every probe, unfiltered: a single probe only surfaces
    # documents it happens to match, which would leave the other audiences'
    # documents missing from the dump and reading like they were not ingested.
    tags: dict[str, set[str]] = {}
    for query in PROBES.values():
        for hit in await store.search(query, top_k=100):
            tags.setdefault(hit.title, set()).add(
                hit.metadata.get("audience", "(untagged)")
            )

    roles = sorted(config.auth.roles)
    failures: list[str] = []

    print("\nrole                audiences                     " + "  ".join(
        f"{a:>9}" for a in PROBES
    ))
    print("-" * 96)

    for role in [None, *roles]:
        granted = config.auth.audiences_for(None if role is None else frozenset({role}))
        label = role or "(anonymous)"
        cells = []
        for audience, query in PROBES.items():
            hits = await store.search(query, audiences=granted, top_k=5)
            # A hit only counts as reaching this audience if the chunk that came
            # back is actually tagged with it -- a public chunk surfacing for a
            # staff probe is a weak match, not a leak.
            reached = any(h.metadata.get("audience") == audience for h in hits)
            expected = audience in granted
            ok = reached == expected
            if not ok:
                failures.append(
                    f"{label}: audience {audience!r} "
                    f"{'leaked' if reached else 'unreachable'} "
                    f"(granted={sorted(granted)})"
                )
            cells.append(f"{'YES' if reached else ' - ':>9}" if ok else f"{'FAIL':>9}")
        print(f"{label:<19} {','.join(sorted(granted)) or '(none)':<29} " + "  ".join(cells))

    if not quiet:
        print("\ndocument tags in the index")
        for title, found in sorted(tags.items()):
            print(f"  {title:<36} {sorted(found)}")

    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("\nOK: every role retrieves exactly the audiences it is granted.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="project id, e.g. shop")
    parser.add_argument("-q", "--quiet", action="store_true", help="skip the tag dump")
    args = parser.parse_args()

    if not Path(f"projects/{args.project}/config.yaml").exists():
        sys.exit(f"no such project: {args.project!r}")

    sys.exit(asyncio.run(main(args.project, args.quiet)))
