# Decision 002 — Why `normal_vegetation` does not belong in the YOLOv8 class definition

**Date:** 2026-07-29  
**Phase:** 3 — Detection and thermal analysis  
**Status:** Resolved → classes reduced to `{flame, smoke}`, false positives handled by negative samples

---

## Context

The original architecture specified three YOLOv8 detection classes: `{smoke, flame, normal_vegetation}`. The third class was intended to suppress false positives — to give the model an explicit concept of "this is ordinary forest, not a fire."

While planning the Phase 3 training data pipeline, I looked at what data would actually be needed to train that third class, and the question turned into a different one: what does `normal_vegetation` do inside a detection model at all?

---

## What I observed

Two things, and the second is decisive.

**Detection already models background implicitly.** In YOLO training, every region *not* covered by an annotation box contributes to "no object here" in the loss. The statement "normal vegetation is not fire" is already being learned by construction. An explicit class adds no negative-discrimination capacity the architecture doesn't already have.

**A bounding box for `normal_vegetation` has no definable extent.** Flame has spatial boundaries. Smoke has approximate ones. Normal vegetation *is the whole frame*. Any box drawn for it would be arbitrary in extent — high annotation noise by construction — disproportionately large in the loss, and would produce meaningless "vegetation detected" boxes at inference that then need filtering out downstream.

There is also a capacity cost specific to this project. YOLOv8n was chosen precisely because of the Jetson Nano inference budget, so model capacity is the scarce resource. Spending part of it on a class that shouldn't exist takes it from `smoke`, which is both the hardest class to learn and the most operationally valuable one — smoke appears earlier and is visible from further away than flame, which is the entire basis of the early-warning claim.

---

## Root cause analysis

The class isn't wrong in itself — it is **misplaced**.

For a whole-frame **classification** task, `{fire, smoke, normal}` is a correct and standard formulation. That is exactly the shape of FLAME 2's labels. The three-class list appears to have been carried over from a classification framing and applied unchanged to a **detection** framing, where it stops being valid.

This is structurally the same failure as Decision 001. There, ORB was a tool designed for single-modality matching, applied cross-modally, producing plausible-looking but meaningless output. Here, a class scheme designed for classification was applied to detection, producing a class that can be trained but cannot be annotated coherently. In both cases the failure is silent: nothing crashes, the output looks reasonable, and the mistake is only visible if you ask what the operation actually means in the framework you're using.

**Information availability makes the same point.** Images of normal vegetation are abundant — tens of thousands of NoFire frames in FLAME 2 alone. Bounding-box labels for normal vegetation are not scarce; they are undefined. No dataset provides them because the annotation has no meaning. When raw material is free but annotation is conceptually impossible, the concept doesn't belong in the annotation schema.

---

## Decision

**Class definition reduced to two classes:**

```
BEFORE:  {smoke, flame, normal_vegetation}
AFTER:   {flame, smoke}
```

**False-positive suppression moves to negative-sample images.** Training images with no annotation file (or an empty one) act as pure negatives. This is the mechanism the framework actually provides for this purpose. Source material is free: FLAME 2 frames labelled NoFire can be used directly with zero annotation work.

**Hard negatives are prioritised over generic ones.** Images specifically prone to being misread as fire or smoke are worth more than bulk ordinary forest imagery:

| Misread as | Sources to collect |
|---|---|
| `smoke` | fog, morning mist, low cloud, dust, water vapour |
| `flame` | autumn foliage, vegetation under sunrise/sunset light, specular leaf glare, red rooftops and vehicles |

Fog versus smoke is the highest-value pair. The two are visually close, and morning fog is routine in mountainous terrain — a system that alarms on every foggy dawn is not deployable. This deserves deliberate collection rather than incidental coverage. The list above is generic and should be revised against what actually occurs in the target deployment environment.

**Acceptance criterion restated.** The original Phase 3 criterion no longer has a referent:

```
BEFORE: False positive rate (vegetation class) < 5%
        → "vegetation class" no longer exists; unmeasurable as written

AFTER:  False positives per image on a held-out negative set < 5%,
        reported SEPARATELY for:
          (a) general negatives — ordinary aerial vegetation, no fire
          (b) hard negatives    — fog/mist, autumn foliage, sunset-lit vegetation
```

Reporting the two separately is deliberate. Performance on ordinary vegetation is a baseline expectation; performance on fog and autumn foliage is the real measure. One combined number would average away exactly the weakness that matters.

---

## Consistency with the wider architecture

This makes the RGB path consistent with the philosophy already adopted on the thermal path.

Thermal Stage 1 uses a recall-first threshold with a hard 60 °C floor and deliberately delegates false-positive suppression downstream to Stages 2/3 and to RGB fusion. The same logic now applies to RGB: **YOLOv8's job is to find flame and smoke with high recall. False-positive suppression is not achieved by adding a class — it comes from negative training data plus the downstream IoU fusion and decision engine.**

The two modalities' weaknesses are complementary, which is the actual justification for fusing them:

| Path | Primary false-positive source | Why the other path catches it |
|---|---|---|
| Thermal | Sun-heated hard surfaces (asphalt, dark rooftops) | No flame texture in RGB |
| RGB | Fog, autumn foliage | No thermal signature |

Neither path should be expected to carry the full false-positive burden alone. Loading it onto one modality would defeat the point of having two.

---

## What this taught me

The pattern worth naming, because it has now happened twice: **right answer, wrong framework.** A concept that is coherent in one framework gets reused unchanged in another where its meaning silently evaporates. Decision 001 was ORB from single-modality to cross-modality. This one is a class scheme from classification to detection.

Neither was caught by the code failing. Both required asking what the operation means rather than whether it runs.

---

## Contribution log

**My reasoning:** raised the question of what `normal_vegetation` contributes and whether the data for it is obtainable; the reassessment came from questioning an assumption I'd carried unexamined from the original architecture rather than from a failure forcing it.

**AI assistance:** articulated the implicit-background mechanism in detection loss; surfaced the negative-image / hard-negative distinction as the standard replacement; drafted the acceptance-criteria restatement.

**Open for my own follow-up:** the negative-sample fraction is currently unspecified. I should determine it from my own validation results rather than adopting a conventional default, and record the number and its justification here. The hard-negative list also needs revising against local conditions.
