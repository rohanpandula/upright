# Contributing

The most useful contribution is closing the family-selection gap. Everything else is smaller.

## The open problem

On real photographs the port picks the wrong line families, so `Full` loses to plain identity
40.7% of the time. With oracle family selection the `Full` median error falls from 3.66% to 0.93%.
A VP-based selector was tried in round 12 and closed none of that gap, so it was reverted; the
evidence is in `evidence/round12.txt` and the ceiling in `evidence/round11.txt`.

Constraints that matter when working on it:

- Detection is adequate. Do not change the detector to fix selection.
- Do not tune the quality gate to hide bad solves. Gate and accuracy are separate.
- The regression corpus is `evidence/round10_*.txt` and the four `lrtest/in` scenes. Nothing may
  regress on either.
- Reproduce a failure before you edit. Every round in `evidence/` does this.

## Ground rules

**Provenance tags.** Every constant carries `PROVEN`, `MEASURED` or `CHOICE` in the port's README.
If you add one, tag it. `CHOICE` is an honest tag, not a weakness; three exist already. An
untagged constant will be sent back.

**Numbers come from runs.** If you claim a figure, point at the file that produced it. The
round files are the model for this.

**Tests.** `cd port && python3 -m pytest -q` must be green. Add a test that fails before your fix
and passes after.

**No Adobe source.** The port is clean-room. The binary is read for constants and control flow
only. If you lift something from a decompiler, say so.

## What is not wanted

- A dependency that is not numpy or OpenCV without a reason.
- Rewriting the energy to a form that scores better on the four scenes while failing the
  real-photo corpus. The scenes are small and unrepresentative; that is exactly how rounds 5 and 6
  went wrong.
- Tidying that moves files the tests resolve by relative path.

## Submitting

Open an issue with the failing case first if you can. A patch with a reproducer and a
before/after number gets reviewed quickly. A patch with neither usually comes back as questions.
