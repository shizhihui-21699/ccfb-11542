# Ring Detection

Offline analysis for coupon abuse where every individual account looks fine but
the accounts share devices, cards, addresses and IPs. It links accounts through
their shared resources, cuts the resulting graph into groups, scores each group,
and shows a reviewer the evidence behind every score.

- **Incremental**: import one day's file at a time; re-importing a file is a
  no-op, and a corrected re-export replaces that day exactly.
- **Scale**: 1M accounts / 3.5M rows / 30 days builds in ~9 s.
- **Explainable**: every ring carries its shared rare entities, its strongest
  account–account ties, and the entities that link it to other rings.
- **Validated**: a generator plants rings *and* legitimate co-tenants, so both
  recall and false-positive rate are measurable (`ringdetect evaluate`).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
ringdetect generate --out data/sim --accounts 20000 --days 30 --rings 8
ringdetect import   --state state/state.db 'data/sim/daily/*.csv'
ringdetect build    --state state/state.db --out results/
ringdetect serve    --out results/ --state state/state.db --open
```

(The package can also be run without installing, as `python -m ringdetect.cli …`
from the repository root. `pip install -e .` just gives you the short name.)

No network access is needed at runtime: the viewer vendors its graph library and
the API is `http.server`. The only non-standard dependencies are numpy, igraph
and leidenalg.

## How it works

### 1. The graph is bipartite, the analysis is on accounts

Raw data is `(account, day, device, card, address, ip)` rows. Accounts and
resources form a bipartite graph. Rings are clusters of **accounts**, so the
graph is projected onto the account side.

### 2. Shared resources are weighted by how rare they are

For an entity `e` used by `d` accounts out of `A`:

```
rarity(e)     = 1 - ln(d) / ln(A + 1)          # 1.0 if only two accounts have it
persistence   = n_days / (n_days + tau)        # long use counts for more
strength(a,e) = sqrt(type_weight[e] * rarity(e) * persistence(a,e))
pair(a,b)     = 1 - prod(1 - evidence)         # noisy-OR over shared entities
```

`rarity` is what stops a shopping-centre Wi-Fi IP or a popular device from
linking thousands of strangers. Per-type trust defaults to device `1.0`,
card `1.0`, address `0.55`, IP `0.40` and is overridable.

### 3. Address and IP corroborate but never create an edge

This is the key modelling decision. Two unrelated households can share a
street address; a whole ISP pool shares IPs. If those formed edges on their own,
co-tenants would be welded into fake 10-account rings — which is exactly the
false positive that makes analysts stop trusting a system. So:

- **device / card** → strong channel, creates edges
- **address / IP** → weak channel, *only adds weight to an edge that strong
  evidence already links*

Entities shared by more accounts than `--pair-degree-cap` (default 300) are
treated as hubs: they are reported as context but generate no edges at all.

### 4. Leiden with CPM, not modularity

Communities are found with **Leiden** optimising the **Constant Potts Model**.

Modularity has a resolution limit: on a graph with hundreds of thousands of
communities it cannot separate a 20-account ring from the giant component
around it, so one threshold cannot work at both small and large scale. CPM
maximises `sum_c (m_c - gamma * C(n_c, 2))` — it keeps a group together while
its *average internal pair weight* exceeds `gamma`. That is a density
threshold, so the same `gamma` means the same thing at 10 accounts and at 10
million. Leiden additionally guarantees every community is internally
connected, which matters when each one is presented as evidence.

`gamma` is the one knob worth tuning: `0.03` (default) keeps large rings whole;
raising it yields tighter, smaller, cleaner groups for triage.

### 5. Scoring separates cartels from co-tenancy

A fraud ring recycles a *cartel*: several devices and several cards. A
legitimate household shares exactly **one** device and **one** card no matter
how many people live there. That structural difference is the main signal:

| component | meaning |
|---|---|
| **cartel** | distinct rare devices/cards shared by ≥2 members, gated on a count ramp (2 → 0, 5 → 1) and on *core fraction* (share of members in ≥2 shared entities) |
| size | group size, saturating |
| closure | share of the group's edge weight that stays inside |
| burst | registration concentration, ranked *within its size band* — a burst is only unusual relative to comparable groups |
| density | mean internal pair weight |
| weak_evidence | address/IP support |

The cartel component dominates. On the 1M fixture its distribution for rings
and households does not overlap at all: households sit at 2 distinct shared
strong entities, rings at 4–8.

Rings whose evidence is address/IP only are marked **weak** and shown
differently: they are a much weaker claim than a shared device.

## Measured behaviour

1M accounts, 30 days, 60 planted rings, 3000 legitimate households:

| metric | result |
|---|---|
| planted rings fully recovered (≥95% of members together) | **56 / 60** |
| planted rings found as a ring scoring ≥ 40 | **60 / 60** |
| households flagged in a ring scoring ≥ 40 | **0 / 3000** |
| households flagged at ≥ 60 | **0 / 3000** |
| build time (projection + Leiden + evidence + write) | **~9 s** |

```bash
ringdetect evaluate --out results/ --manifest data/sim/manifest.json
```

Reproduce with a bigger fixture:

```bash
ringdetect generate --out data/big --accounts 1000000 \
    --days 30 --rings 60 --households 3000 --gzip
```

## Daily import

```bash
# one file per day; the date column (or --day) supplies the day
ringdetect import --state state.db data/daily/events-2026-09-01.csv
ringdetect import --state state.db --day 2026-09-02 data/daily/export.csv

# a corrected re-export of a day: replace, don't add
ringdetect import --state state.db --replace-day 2026-09-01 corrected.csv

# different column names / a raw PAN that must be hashed
ringdetect import --state state.db --account-col uid --device-col imei \
    --card-salt-env CARD_SALT data.csv
```

Import is content-addressed by SHA-256, so re-running a file is a no-op, and
`--replace-day` deletes the old ledger for a day before re-reading so a fixed
file is exact rather than additive. Raw card numbers are refused unless a salt
is configured; with one they're stored as a salted HMAC prefix.

Schema, per day: `link_day` is the per-day ledger, `link` the rolled-up
account→entity counters the build reads, `batch` the import history.

## Result set

`build` writes `results/results.sqlite` (rings, members, evidence entities,
evidence edges, ring-to-ring bridges, parameters) and `summary.json`. The
result directory is **self-contained** and portable — copy it to an analyst's
laptop and `ringdetect serve --out it/` works with no state database (the
bipartite network view needs `--state` as well; without it the viewer falls
back to account-to-account edges).

## Viewer

```bash
ringdetect serve --out results/ --state state/state.db --port 8765
```

- ranked ring list with a score histogram you can click to filter
- per-ring network: accounts plus the shared devices/cards/addresses they
  actually have in common (entities shared by only one member are not drawn)
- tabs for shared entities, accounts, strongest ties, and links to other rings
- every account is searchable across rings; every entity value is copyable
- light and dark themes; works at phone width

The API is read-only and same-origin:

```
GET /api/summary          build metadata + score histogram
GET /api/rings?min_score=50&sort=score&limit=50&offset=0
GET /api/rings/<id>       members, evidence entities, edges, bridges
GET /api/rings/<id>/graph nodes/edges for the network view
GET /api/accounts/<key>   rings for one account
GET /api/batches          import history (needs --state)
```

## Tuning

| flag | default | effect |
|---|---|---|
| `--gamma` | `0.03` | CPM resolution. Higher = smaller, tighter rings. |
| `--pair-degree-cap` | `300` | entities used by more accounts than this create no edges |
| `--min-score` | `0` | drop rings below this score at build time |
| `--type-weight device=0.2` | — | repeatable; per-type trust |
| `--ip-prefix 24` | `0` | collapse IPv4 to /24 to blunt NAT noise |

## Limitations

- The generator's rings are cleaner than reality: one operation may be split
  across several detected rings (the *Linked rings* tab surfaces the entities
  that would merge them), and a ring's members may be partially recovered.
- Rings joined only by a shared address or IP are marked weak rather than
  dropped; whether to act on them is a policy decision, not a scoring one.
- Entity values are stored as given after normalisation. Device fingerprints
  that change on app reinstall, or addresses typed slightly differently, will
  fail to merge — that is a data-quality issue upstream, not something the
  graph can recover.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # 72 tests, ~2.5 s
```

Covers normalisation and card hashing, day parsing, idempotent and
corrective imports, projection-weight properties (rarity, hub caps,
strong-vs-weak channels), community separation, the household-vs-ring scoring
claim, the full CLI pipeline, and every HTTP endpoint including path traversal.
