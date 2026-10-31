# ImportSweep — operating the Upright sweep harness

`ImportSweep.lua` drives Lightroom Classic's own Upright solver over a folder of
test images and records every step to a TSV trace. It exists because a naive
version failed silently in nine different ways, each of which cost hours.

**Invariants — do not change:**

| thing | value |
|---|---|
| menu item | `Library > Plug-in Extras > Import Sweep` (title lives in `Info.lua`) |
| trace file | `lrtest/TRACE.txt` |
| lock file | `lrtest/.ImportSweep.lock` |
| scratch dir | `lrtest/.sweep-tmp` |

`Init.lua` is **never executed** by Lightroom Classic. Only the menu item runs.
A copy of this harness in `uprightprobe_nm.lrdevplugin/` is separate and writes
`TRACE_NM.txt`; it does not take the lock and is not covered here.

---

## 1. What a run does

1. Takes the lock (§4). Refuses to start if another run holds it.
2. Dismisses a modal "internal error" alert if one is up.
3. Checks the library filter and clears it if active.
4. Imports the images, puts them in a **new collection**, makes that collection
   the **active source**, and selects the first photo.
5. Checks that the right panel is not collapsed (fixes it if it is).
6. **Probe:** selects a photo and asserts `getTargetPhoto()` really returns it.
   Aborts if not — before doing any slow work.
7. For each image × mode: reset to mode 0, arm the target mode with a zeroed
   digest, press the panel **Update** button, poll for a new transform.
   On an empty result: re-select and retry once, then record `FAILED`.
8. Writes a `SUMMARY`, releases the lock.

Two flows exist:

- **sweep** (default) — the folder sweep above, modes from `UPRIGHT_MODES`.
- **job** — used by the guided-limits runner: reads `/tmp/uplim/job.txt`
  (`SETUP|collection|path…` then `CASE|id|path|mode|count|priorAuto|segments…`),
  writes `results.txt`, and waits for a `CLICKED` marker. If no clicker creates
  that marker within 6 s it presses Update itself, so it works unattended.

Force one with `UPRIGHT_FLOW=sweep` or `UPRIGHT_FLOW=job`.

---

## 2. Preconditions

Before invoking the menu item:

- [ ] Lightroom Classic is open on the **catalog that already contains the test
      images**, and nothing else is driving the UI.
- [ ] No other agent is mid-run: `lrtest/.ImportSweep.lock` absent, or older
      than 45 min.
- [ ] macOS has granted Accessibility to Lightroom **and** to whatever runs
      `osascript` (System Settings → Privacy & Security → Accessibility).
      Without it every AX call fails and the trace fills with
      `ALERT_WATCHDOG_ERROR` / `PANEL_UNKNOWN`.
- [ ] Window is on-screen and reasonably wide. `Window > Fill` is applied
      automatically if the right panel looks collapsed, but a window dragged
      fully off-screen still defeats OCR-based clickers.
- [ ] Library filter is off. The harness clears it, but a filter that keeps
      re-applying (e.g. an active text search) shows up as `FILTER_STUCK`.
- [ ] You are in the **Library** or **Develop** module. The harness switches to
      Develop itself.
- [ ] Test images exist in `lrtest/abstain` (default) or `UPRIGHT_DIRS`.

The harness refuses to solve when the probe fails. A run that produces no rows
after `PREFLIGHT_FAIL` is working as intended — read the `detail` column.

---

## 3. Why the digest must be zeroed (critical)

A matching `UprightDependentDigest` **suppresses the solve**. `applyDevelopSettings`
does not itself trigger anything: writing `PerspectiveUpright` alone is silent,
and a stale matching digest makes Lightroom believe the existing transform is
still valid.

`enableUprightReanalyze` recomputes only when `mode != 0` **and** the stored
digest differs from the digest of the current geometry inputs. The harness
therefore always:

1. writes `PerspectiveUpright = 0` (mode 0 never recomputes — a clean baseline),
2. writes the target mode **plus 32 zeros** into the digest
   (`UprightGuidedDependentDigest` for guided, `UprightDependentDigest` otherwise),
3. presses the panel **Update** button — the explicit re-analyse.

Zeros never compare equal, so the Update always solves. This removes any need to
hand-draw guides or click mode buttons for a re-solve.

Guided (mode 5) additionally requires a non-empty segment vector, otherwise the
predicate short-circuits to "no recompute" before it even looks at the digest.
`UPRIGHT_GUIDES` supplies segments; the built-in default is a placeholder frame
that satisfies the gate but is not scene-specific.

After arming, the harness **reads the digest back**. If the write did not stick
it says `digest=STALE` in the `ARM` row and falls back to matrix/focal change
detection. `ARM … digest=STALE` in a trace means the arm was weak — do not trust
`solved=yes` rows from that run without checking the transforms.

---

## 4. Lock protocol

`lrtest/.ImportSweep.lock` holds `run=`, `agent=`, `ts=`, `iso=`.

- **Acquire:** if the lock exists and is younger than 45 min → `LOCKED` +
  `ABORT`. Older → `LOCK_STALE_STEAL` and take it.
- **Race:** after writing, the owner re-reads the lock; if a foreign run id is
  there, it stands down (`LOCK_LOST`).
- **Heartbeat:** rewritten after every image. If someone else took it mid-run,
  `LOCK_STOLEN` is recorded and the run continues (rows stay separable by
  `run_id`).
- **Release:** removed only if the lock still carries this run's id.

Two agents can therefore interleave traces safely, but they will still fight
over the UI. Prefer one at a time.

---

## 5. Trace format

TSV, tab-separated, **15 columns, fixed order.** The header row is written only
when the trace file is empty (`traceHeaderIfNeeded`, `ImportSweep.lua:216-228`,
called once before `RUN_START`). An existing non-empty trace never gets one, so
the current `lrtest/TRACE.txt` has **no** header row. Add it by hand if you want
it — the exact row is:

```
run_id  ts  agent  event  image  mode  attempt  target  solved  digest  f35  panel  filter  alert  detail
```

`-` means empty. Tabs/newlines inside values are flattened to spaces. Read it
with `awk -F'\t'` or `csv.DictReader(delimiter='\t')`.

The first five lines of `TRACE.txt` are older `RunProbe.lua` output
(`23:22:40  === digest retest …`), not TSV, so neither recipe above parses the
file from line 1. Select TSV rows with `$1 ~ /^[0-9]{8}T/`.

`run_id` is `<ISO8601>-<epoch>-<rand>`; `agent` is `$UPRIGHT_AGENT` or `$USER`.
Filter a run with `awk -F'\t' '$1=="<run_id>"'`.

### Event reference

| event | meaning |
|---|---|
| `RUN_START` / `RUN_END` | run boundaries; `RUN_END detail=error` means an uncaught error |
| `FLOW` | `job` or `sweep` |
| `LOCK_OK` / `LOCKED` / `LOCK_STALE_STEAL` / `LOCK_LOST` / `LOCK_STOLEN` / `LOCK_RELEASED` | lock lifecycle |
| `ALERT_DISMISSED` / `ALERT_WATCHDOG_ERROR` | modal dismissed / AX call failed |
| `PREFLIGHT_BEGIN` / `PREFLIGHT_OK` / `PREFLIGHT_FAIL` | preflight |
| `PANEL_STATE` / `PANEL_COLLAPSED` / `PANEL_FIXED` / `PANEL_FIX_FAILED` / `PANEL_UNKNOWN` | right-panel condition |
| `FILTER_ACTIVE` / `FILTER_CLEARED` / `FILTER_STUCK` | library-filter condition |
| `FOUND` / `MISSING` / `IMPORT` / `COLLECTION` / `ACTIVE_SOURCE` | import and source activation |
| `TARGET_OK` / `TARGET_NIL` | selection self-check (identity, not just non-nil) |
| `ARM` / `WRITE_FAILED` / `UPDATE_PRESSED` / `UPDATE_MISSING` | arming and the re-analyse trigger |
| `SOLVE_OK` / `SOLVE_EMPTY` / `RETRY` / `FAILED` | per-attempt outcome |
| `IMAGE_DONE` / `SUMMARY` / `ABORT` | progress and job-flow markers |
| `JOB_BEGIN` / `JOB_SETUP` / `JOB_ARMED` / `JOB_MISSING` / `JOB_TIMEOUT` / `JOB_STALE_MARKER` / `JOB_END` | job-flow markers; `JOB_TIMEOUT` means no transform appeared for that case |

### Failure signatures → cause → fix

| signature | cause | fix |
|---|---|---|
| `TARGET_NIL … first attempt` then `… after-source-reassert: nil` | photo not in the active source, or a filter hides it | check `FILTER_*` rows; re-import; make the collection active before selecting |
| `FILTER_ACTIVE` with `0 of N photos`, then `FILTER_STUCK` | filter bar still filtering after "None" | clear the filter by hand (`Cmd-L`, click **None**, or `Library > Filter`), rerun |
| `PANEL_COLLAPSED` then `PANEL_FIX_FAILED` and `ABORT` | right panel collapsed and `Window > Fill` + `Cmd-Shift-3` did not restore it | widen the window manually, expand the right panel, rerun |
| `PANEL_UNKNOWN` + `ALERT_WATCHDOG_ERROR` | Accessibility permission missing, or Lightroom not frontmost | grant Accessibility, bring Lightroom forward, rerun |
| `ALERT_DISMISSED` repeatedly | solver crashing on those images | note the images, expect `FAILED` rows, inspect them separately |
| `ARM … digest=STALE` | digest write no-op (busy catalog / write lock) | rerun that image; if it persists, check for another writer on the same photo |
| `SOLVE_EMPTY` twice then `FAILED` | solver produced nothing (blank/flat scene, mode 5 without usable guides, digest not invalidated) | inspect the image; supply real `UPRIGHT_GUIDES`; confirm the arm row was not `STALE` |
| `UPDATE_MISSING` | Update button not in the AX tree (wrong module, panel scrolled away) | ensure Develop + Transform/Upright visible; rerun |
| `LOCKED` | another run holds the lock | wait, or delete the lock if the owner is known dead |
| `ABORT … uncaught` | Lua error | read the message; the lock is released on the way out |

---

## 6. Running it

Manual: `Library > Plug-in Extras > Import Sweep`.

Environment (set before Lightroom launches — Lightroom inherits the shell env):

| var | default | meaning |
|---|---|---|
| `UPRIGHT_DIRS` | `lrtest/abstain` | colon-separated image folders |
| `UPRIGHT_MODES` | `1,2` | modes: 1 auto, 2 full, 3 level, 4 vertical, 5 guided |
| `UPRIGHT_GUIDES` | placeholder frame | `x1,y1,x2,y2;…` for mode 5 |
| `UPRIGHT_LIMIT` | `0` (all) | cap on image count |
| `UPRIGHT_TIMEOUT` | `30` | seconds to wait for one solve |
| `UPRIGHT_AGENT` | `$USER` | label in the `agent` column |
| `UPRIGHT_FLOW` | `auto` | `sweep`, `job`, or `auto` (job file present → job) |
| `UPRIGHT_JOB` | `/tmp/uplim/job.txt` | job-file path |

Post-run: check `SUMMARY` (`failed=0` is the only clean result), then read
transforms out of the catalog as usual.

---

## 7. Non-goals

- No export and no image analysis — it only harvests develop settings.
- No clicking by OCR/coordinates; it uses the AX tree only. If AX is blind, it
  aborts rather than guessing.
- It does not clear old collections; each run makes `sweep <run_id>`.
- It does not undo its own writes; the reset-to-0 at the start of each attempt is
  the only normalisation.

---

## 8. File ownership / concurrency

`ImportSweep.lua` is shared with the guided-limits work, which briefly replaced
it with a `/tmp/uplim`-only variant and moved the trace. The current file keeps
the `lrtest/TRACE.txt` destination and supports both flows, so **restore from
here rather than from a `/tmp/uplim` copy**:

```
cp lrtest/ImportSweep.lua.bak lrplugin/uprightprobe.lrdevplugin/ImportSweep.lua
```

That backup is **historical, not current**. Both files carry the same `BASE` and
both pass `luac -p`, so the backup still loads — but `diff` against the shipped
file is **160 lines**, and restoring it drops everything added since: the
`getenv`/`removeFile` sandbox wrappers (`os.getenv` and `os.remove` do not exist
in this Lua build, so without the wrappers every env var reads nil), multi-window
AX lookup for Transform/Update/Upright, the identity-homography slot wipe, guided
segments written in a separate prior write, and the digest-restamp early-out.
Excerpt, `.bak` on the left:

```
28a29,43
> -- os.getenv is absent in the LR Lua sandbox on this build; wrap it.
> local function getenv(k)
>     local ok, v = pcall(function() return os.getenv(k) end)
>     if ok then return v end
>     return nil
> end
45c60
< local JOB = os.getenv('UPRIGHT_JOB') or '/tmp/uplim/job.txt'
---
> local JOB = getenv('UPRIGHT_JOB') or '/tmp/uplim/job.txt'
```

Refresh the backup from the shipped file before relying on it, and record the
date it matches. Older copies in `/tmp/uplim/ImportSweep.orig.lua` and
`/tmp/sweeptest/` predate job mode and are for history only.

If two agents must run at once, stagger them: the lock serialises trace writes
but not UI focus, and AX presses from one run can land in the other's dialog.
