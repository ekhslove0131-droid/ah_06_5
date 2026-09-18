# CBJ Stack-7 train-only probability bank

Prepared locally and not launched. This package reconstructs a fixed seven-model
training bank for a later, separately reviewed stacking search.

This package never adopts an unbound existing bank NPZ. Crash recovery is permitted only
when `TRAINING_BANK.PENDING.json` already binds the exact NPZ hash, producer
source/config identity, lineage and array hashes. A matching pending binding can
restore a missing final receipt; otherwise the existing artifact is retained
and preparation fails closed.

V3 applies the same rule to partial publication boundaries. An existing
`.TRAINING_BANK.npz.partial` is adopted only when an already durable pending
receipt binds its exact hash and producer identity. An existing pending-receipt
partial is compared byte-for-byte and adopted only when it matches; foreign or
stale partial bytes are retained and rejected, never overwritten.

It writes two probability tensors only:

- inner OOF: `7 × 4,959 × 26`
- full five-fold OOF: `7 × 6,201 × 26`

The order is frozen as raw `ANCHOR-H1`, raw `ANCHOR-C10`, `OLD-G3-0014`,
`OLD-G3-0018`, `TEAM-T2-0005`, `TEAM-T2-0007`, `TEAM-T4-0007`.

Inner candidate probabilities come from the pinned V2 evidence. Raw H1/C10
are reconstructed from the 15 already verified historical anchor receipts with
the original outer-0 inner namespaces. Full OOF comes only from the `valid`
member of the completed V3 35-component cache. The original NPZ is hashed in
full, but its `test` member is never loaded, selected or copied.

`preflight` is read-only and reports exact missing or mismatched artifacts.
Historical anchor artifacts are read through a dedicated non-creating view, so
a missing cache cannot create directories during preflight. The graded base
source must equal the literal frozen `b56d607e...` identity as well as all
prepared/config/V3 receipts.
`prepare-bank --start` is required before any output is written. This package
does not fit a model, launch a stacking grid, read test predictions, perform a
full refit or submit to a website.
