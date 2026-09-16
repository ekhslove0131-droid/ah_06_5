# CBJ completed combination recovery V3

This version preserves the completed V2 score ledgers and their original
source/config identities. `completed_replay` verifies all three complete grids,
copies immutable score records and bias caches into a separate run, and
reconstructs selected graphs without scoring new combinations. Equivalent
single-source pooling aliases replay the representative computation that was
actually scored. It does not weaken probability-hash reproduction checks.

For the approved recovery, run `search --config runtime_config.windows.json
--start` directly. The existing V2 evidence is read-only and already frozen;
do not run `prepare-evidence` into that existing evidence directory. Deployment
uses the same frozen runtime config after the recovered search is complete.

Prepared next-stage package. It does not launch while the graded V2 worker is
active and does not fit a new base-candidate grid.

This revision keeps the V1 grid unchanged and adds write-once recovery for
evidence, parent, frozen-recipe and completed-deployment boundaries. Stage
probability reproduction permits only <=1e-12 numerical drift with identical
argmax predictions; both probability hashes are retained.

## Sequence

1. `preflight`: require the V2 worker to be absent and all 124 candidates to
   have exact three-fold inner OOF caches.
2. Verify the pinned existing evidence and completed-replay source/config hashes.
3. `search --start`: adopt all completed C0/C1/C2 score records and reconstruct
   selected graphs. `ORIGIN.json` preserves the original score producer.
4. Review `SEARCH_COMPLETE.json` and `FROZEN_RECIPE.json`.
5. `deploy --start`: import checksum-matched V2 component predictions, fit only
   missing active components, create adaptive five-fold OOF and a CSV.

The score grid has at most 6,241 declared rows: C0 3,721 plus two 1,260-row
six-parent meta rounds. Every declared row is retained. Exact-equivalent rows
receive alias receipts. There is no minimum-F1 admission
gate and no early grid truncation.

Search reads only outer-0 inner OOF labels and probabilities. Test is opened
only by frozen deployment. Adaptive full OOF is reported after deployment and
does not change weights, bias or the selected recipe.
