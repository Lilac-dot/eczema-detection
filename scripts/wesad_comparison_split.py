"""Shared, fixed patient-wise split for the Stage A (stress) architecture comparison --
every model in this comparison (classical, LSTM, PaPaGei-based) imports these same
constants so they're all trained/validated on the exact same subjects, and none of them
ever load TEST_SUBJECTS. Computed once (seed 42, held-out fraction ~27%, same proportions
used for the moisture-sensor comparison) and hardcoded here rather than re-randomized per
script, so there's no risk of drift between scripts.

TEST SET ISOLATION: TEST_SUBJECTS is not touched by any training script. It is reserved for
a single future evaluation, not to run until explicitly requested (see
docs/transfer_cnn_test_set_access_2026-09-18.md for why this discipline exists -- same
lesson, applied here from the start instead of after a near-miss).
"""

ALL_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
                "S13", "S14", "S15", "S16", "S17"]

TEST_SUBJECTS = ["S11", "S14", "S16", "S2"]
TRAIN_SUBJECTS = ["S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S13", "S15", "S17"]

assert set(TRAIN_SUBJECTS) | set(TEST_SUBJECTS) == set(ALL_SUBJECTS)
assert set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS) == set()

# For internal validation during training (early stopping / epoch monitoring) -- a further
# subject-wise split of TRAIN_SUBJECTS only. Never touches TEST_SUBJECTS.
INTERNAL_VAL_SUBJECTS = ["S6", "S13", "S17"]
INTERNAL_TRAIN_SUBJECTS = [s for s in TRAIN_SUBJECTS if s not in INTERNAL_VAL_SUBJECTS]
