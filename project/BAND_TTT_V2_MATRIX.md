# Band-TTT v2: validation-first experimental matrix

Status: protocol/design, five-fold validation gradient audit (1280 sampled observations), and real-model exact 1/3/5-step global/layered inner-loop smoke checks completed. Full matrix training/evaluation is not yet implemented or launched. Contextual chronological episode training remains to be integrated.

## Initial measurements

- 128 observations per class per fold, 1280 total, sampled with replacement round-robin across patients; not independent population observations.
- Mean shared-backbone cosine: -0.06156; negative fraction: 55.47%.
- Non-seizure: mean cosine -0.19159, negative fraction 70.31%, classification loss improved in 17.03% of sampled observations.
- Seizure: mean cosine +0.06867, negative fraction 40.63%, classification loss improved in 58.59%.
- SSL loss improved on all sampled observations; this does not imply lower event-level false alarm duration.
- Six real-model differentiable inner-loop checks passed (K=1/3/5 x global/layered), with no source parameter mutation.
- The Band head's separate step size has no classification outer gradient at K=1, as expected: its updated weights are not used in detection. At K=3/5 its step-size gradient exists through subsequent SSL updates.
- recordings.parquet contains no acquisition timestamp. Cross-recording patient chronology must be verified from another authoritative local source before P configurations run.
- Outputs: /root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/gradient_audit_128 and smoke.json. Existing v1 artifacts unchanged.

Primary question: can online Band adaptation reduce false-alarm duration beyond its own frozen checkpoint without unacceptable sensitivity or delay losses?

## Matrix (18 training configurations)

Cartesian product:
- Context: E (independent window, reset each window); R (rolling, reset each recording); P (patient-level state, reset each patient).
- Matched train/test inner steps: 1, 3, 5.
- Learned step sizes: G (global scalar), L (separate scalar for each of the last two encoder blocks and Band head).

IDs: E1G E3G E5G E1L E3L E5L; R1G R3G R5G R1L R3L R5L; P1G P3G P5G P1L P3L P5L.

Every checkpoint is evaluated both frozen and adaptive. Reuse v1 only as a historical reference; run a new matched E1G control. No Temporal experiments.

## Causality and matching

- R/P predict the current query before adding it to adaptation context. Support windows must have end <= query start, avoiding raw-signal overlap across support/query.
- Initial context buffer: 8 eligible preceding windows. Empty context: frozen prediction. No future-window use, label-based update gates or cross-patient state.
- Updates persist inside the specified reset boundary; no replaying the same support indefinitely without a logged update schedule.
- R resets at recording boundaries. P may carry state across recordings only with verified acquisition chronology; filename sorting alone is insufficient. If chronology cannot be verified, P is blocked, not silently approximated by R.
- Match inner steps, support construction, gradient reduction, update frequency and reset policy in training and evaluation. Context-aware meta-training requires chronological episodes, not the old shuffled window sampler.
- Use exact second-order training initially. If memory requires smaller micro-batches, preserve effective outer batch and log the change; never silently switch to first-order.
- Same initial backbone, seed, learning-rate bounds, data split, early-stop budget, Band transform and adaptive parameter subset across matched comparisons.

## Stages

1. Audit existing Band checkpoints on validation only: shared-backbone gradient cosine, blockwise cosine, gradient norms, actual and predicted classification-loss changes. Labels diagnose alignment only and never enter an adaptation update.
2. Implement/test v2 episode loader, multi-step differentiable update and grouped step sizes. Verify K=1 parity, gradients to each step-size scalar, parameter-reset isolation and causality. Audit batch-mean versus independent-sample gradient semantics; do not assume a simple batch-size multiplier fixes it.
3. Screen all 18 configurations on folds 0 and 1, fixed seed 3407 and identical budgets: 36 training runs, 72 paired validation evaluations. These folds are chosen in advance, not based on new test performance. Gradient audits accompany each checkpoint.
4. Extend the locked E1G control plus up to three validation-selected candidates to remaining folds: at most 12 additional training runs (48 total, 96 paired validation evaluations). This stage is exploratory because selection uses existing validation data.
5. Do not launch a fresh test sweep during screening. Freeze candidate selection and protocol first. The v1 test data have already been inspected, so another run on them is exploratory, not independent confirmation. Seek a new held-out cohort or separately authorized confirmatory evaluation.

## Endpoints and decision rules

- Primary: false-alarm minutes/24h, raw seizure intervals excluded, existing eventization/collar definitions unchanged.
- Co-primary guardrails: sensitivity and detection-delay distribution. Report a Pareto comparison; do not reduce alarm burden by hiding missed events or later detection.
- Report both a fixed matched-checkpoint threshold (frozen validation selected threshold) and separately validation-calibrated thresholds; never optimize thresholds on test.
- Validation threshold selection uses the existing sensitivity >=80% target, with feasibility explicitly reported.
- Secondary: FA events/24h, window AUPRC/AUROC (per-fold and pooled), median/p90 delay, drift, updates/hour, latency, peak GPU memory.
- Paired patient bootstrap (2000), confidence intervals and multiplicity-adjusted comparisons. Report all matrix cells, including negative and failed outcomes. Final non-inferiority margins require explicit prespecification before any confirmatory claim.

## Gradient reporting

- Cosine computed over shared adaptive backbone parameters only. SSL-only head is not padded into the classification vector; its gradient norm is reported separately.
- Global and per-block statistics, by fold/patient/class/Band transformation, before adaptation and after inner steps.
- Record zero-norm/unused gradients rather than treating their cosine as 0.
- Record SSL loss reduction, classification-loss delta, -alpha*g_cls dot g_ssl, parameter relative update norm and step-size-weighted alignment.
- The initial audit is balanced diagnostic sampling, not prevalence-weighted performance estimation. It does not alone establish the cause of low event-level gains.
