# Decision 003 — Training data sources, and the UAV × local-vegetation gap

**Date:** 2026-07-29  
**Phase:** 3 — Detection and thermal analysis  
**Status:** Resolved → data sources selected; self-annotation established as a required work package  
**Amended:** 2026-07-29 — see Amendment A. The central finding below is **partially superseded**: a UAV sub-dataset with both classes was subsequently found. Read Amendment A before acting on the data plan in this record.

---

## Context

The plan listed FLAME 2 as the primary YOLOv8 training source without anyone having checked what kind of labels it carries. Planning the Phase 3 data pipeline forced that check, and it failed — which cascaded into a full survey of what public wildfire detection data actually exists.

The survey produced something more useful than a dataset list: a quantified description of what is missing.

---

## What I observed

### FLAME 2 cannot train a detector

FLAME 2's label file (Item #10) is **543 bytes** for 53,541 frame pairs. That size settles it — it cannot contain per-object box coordinates. It encodes only whole-frame binary classifications: Fire/NoFire and Smoke/NoSmoke.

FLAME 2 therefore supports classification and dual-light fusion work, but cannot train the YOLOv8 detector, which needs box-level annotation. Listing it as the primary detector training source was wrong.

The general lesson: **"has labels" is not the same as "has the labels this task needs."** Classification labels and detection labels are different artifacts, and the difference is invisible until you check the label format.

### FLAME 2 download scope (revised)

Full dataset is 160 GB+. The required subset is about 5% of it:

| Item | Content | Size | Decision |
|---|---|---|---|
| #9 | 53,451 frame pairs, 254×254, cropped to matched RGB/IR FOV | 8.3 GB | **Download** |
| #10 | Fire/Smoke frame labels | 543 B | **Download** |
| #11/#12 | READMEs | KB | **Download** |
| #1-7 | 7 raw video pairs | 13.6 GB | Defer — only needed for dA/dt temporal work |
| #8 | Same frames at original resolution | **126 GB** | **Never** — identical content to #9, 15× the size |
| #13–#18 | Burn plan, pointcloud, orthomosaic, weather, preburn video | ~4.3 GB | Skip — remote-sensing products, not applicable |

Item #9's description states the frames were cropped so RGB and IR share a similar FOV and perspective — it is **pre-registered**. Same situation as LLVIP in Decision 001: fusion work proceeds with `H = Identity`.

### The public dataset landscape

**Boreal Forest Fire** (Pesonen et al., *Scientific Data* 2025) is the strongest UAV-perspective detection dataset available. Four Finnish prescribed burns, DJI Phantom 4, 4K RGB, 10–200 m AGL, camera pitch 0° to −90°.

- Subset A: 4,954 images with YOLO-format boxes, single class `smoke` (id 0). 256 of those are empty images shipped with empty annotation files.
- Subset B: 288 4K video clips, per-video smoke/no-smoke labels.
- Subset C: segmentation masks (SAM-generated; 40 hand-drawn as test set).
- CC BY 4.0. DOI `10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9`

Only `smoke` is annotated. The authors note flames appear in much of the footage and suggest extracting and annotating frames for flame detection — so flame boxes remain unsolved here too.

**HPWREN / AI for Mankind**, mirrored on Roboflow at `brad-dwyer/wildfire-smoke`. 737 images, single `smoke` class, CC BY-NC-SA 4.0.

A Roboflow search for "aerial wildfire smoke" returned 300 results, of which at least fifteen report exactly 737 images. Spot-checking confirmed these are re-uploads of this same dataset under different workspace names. **The apparent abundance is an illusion** — the number of genuinely distinct labelled wildfire detection datasets is far smaller than the result count suggests.

HPWREN is a Southern California tower network, which makes it the closest public data to the intended deployment environment, though from a fixed ground perspective rather than UAV. Published metrics on it: mAP@50 92.8%, Recall 94.6%, **Precision 66.0%**. The recall/precision asymmetry is itself informative — even a well-regarded smoke benchmark runs high-recall/low-precision, which is independent support for the recall-first philosophy adopted in Decision 002.

**Kapustin Danil** (`kapustin-danil/wildfire-and-smoke`), 3,255 images, classes `Fire` and `Smoke`, MIT licence. No published description, ~4 years stale. Supplementary at best; annotation quality unverified.

**Rejected example.** `wildfire by Wildfire` (7.25k images) carries a class list including duplicate-case pairs (`fire`/`Fire`), bare numerics (`0`, `1`), a dataset version string used as a class name, a non-English synonym (`duman`), and `candlelight`. Uncurated regardless of scale — recorded here as a concrete instance of the screening criteria.

---

## The central finding

```
Public labelled wildfire DETECTION data, complete inventory:

  HPWREN 737          — local geography ✓ | UAV perspective ✗ (fixed tower)
  Boreal Forest 4,954 — UAV perspective ✓ | local geography ✗ (Finnish boreal)
  Kapustin 3,255      — mixed/unverified, supplementary only

  Intersection of (UAV perspective × local vegetation) = EMPTY
```

This is not an inconvenience to route around. The Boreal paper supplies evidence that the gap **matters** rather than being cosmetic:

> A YOLO v5 L model fine-tuned on HPWREN data reached 0.93 precision on similar
> test data, but only **0.031** on the Finnish Ruokolahti data.

0.93 → 0.031 is not degradation, it is total failure. The authors conclude that locally collected data is required when the background of a semi-transparent object such as smoke differs from the available training data.

**The symmetric implication is the one that applies here.** If HPWREN → Finland collapses, then Boreal → local chaparral carries the same risk. Finnish spruce and pine with lake surfaces and cloud cover versus dry high-contrast chaparral is, if anything, a larger background shift.

Locally-relevant annotation is therefore not optional and not a shortcut taken for lack of better data. It is a documented requirement, and this record is the evidence trail.

---

## Annotation strategy — large boxes

The Boreal authors tested two strategies on the same imagery: large boxes enclosing the whole plume including background, versus multiple small boxes containing only pure smoke.

> YOLO v5 S fine-tuned on 1,630 images: small annotations → precision **0.24**;
> large annotations → precision **0.94**.

They removed the small-box annotations from the published dataset on the strength of that result.

**Decision:** use the large-box strategy when annotating own frames — enclose the whole plume, accept background inclusion, do not trace smoke boundaries. This is counter-intuitive, which is exactly why the experimental result is worth recording rather than trusting intuition. Worth validating on a small local sample before committing, since the Boreal result comes from a different vegetation background.

**Secondary confirmation for Decision 002:** Boreal ships 256 empty images with empty annotation files — the negative-sample mechanism adopted there, implemented in a peer-reviewed dataset.

---

## Licensing

| Source | Licence | Constraint |
|---|---|---|
| HPWREN / AI for Mankind | CC BY-NC-SA 4.0 | **Non-commercial** + share-alike |
| Boreal Forest Fire | CC BY 4.0 | Attribution only |
| Kapustin Danil | MIT | Permissive |

The NC term matters because the comms architecture (P900 / RockBLOCK) was specified as a commercial design target. As a personal and academic project this is unproblematic, but a model trained on BY-NC-SA data inherits the restriction. Recorded now so it is known rather than discovered later.

---

## Decision — revised Phase 3 data plan

```
smoke (primary)   Boreal Forest Fire Subset A — 4,954, YOLO format, ready to use
                  + HPWREN 737 — local geography, ground perspective
flame             not covered by either; source from original FLAME segmentation
                  masks converted to boxes, or annotate directly
negatives         Boreal's 256 empty-annotation images + FLAME 2 NoFire frames,
                  split into general/ and hard/ per Decision 002
local adaptation  self-annotated aerial frames, LARGE-BOX strategy
                  ← established above as required, not optional
fusion / dA/dt    FLAME 2 #9 (pre-registered pairs) and #1-7 (video, when needed)
temperature       FLAME 3 radiometric TIFF — unchanged
```

---

## Related data-handling notes

**FLAME 3 raw plot data needs three preprocessing steps** (from the Hanna Hammock readme; applies to the raw NADIR plot archive, not the CV subset which is already converted):

- Temperature: the TIFF holds raw DN, not Celsius. Convert with `celsius = DN * 0.1 - 275`. Using DN directly as temperature would invalidate every threshold in Stage 1.
- Timestamps run 4 hours behind actual time. A constant offset, so frame-differencing for dA/dt is unaffected; only cross-referencing against externally timestamped data needs correcting.
- Plot 2 is split by a 9 min 10 s pause (parts `826–1147` and `1171–1346`). The temporal tracker must treat these as **separate sequences** — a frame-adjacent growth rate computed across the discontinuity would produce a large spurious dA/dt spike.

That same readme documents the georeferencing pipeline used on real fire-ground data: ECC affine stabilisation against the previous frame, then polynomial(1) georeferencing in QGIS anchored on ground control points. **ECC is the method Decision 001 already nominated for Phase 2**, so this is an independent confirmation of that choice from a working project. The GCP dependency also restates the same principle established for calibration: tying imagery to the real world requires a reference of known real-world geometry.

**Sensor resolution mismatch.** FLAME 2 IR is 640×512; FLAME 3 / Hanna Hammock used an InfiRay-based AUTEL Evo II 640T; Boreal is 4K RGB. Target hardware is the Lepton 3.5 at **160×120**. Validating algorithm logic on higher-resolution data is fine, but any claim about expected on-hardware performance should be made against data downsampled to 160×120.

**`da_dt` is a hardcoded placeholder.** In the Week 2 fusion call it is passed as a literal `5.0`. `IoUFusionEngine.fuse()` only consumes it; nothing in the pipeline produces it. Real dA/dt needs cross-frame tracking of the same hotspot over a time axis, which the static frame-pair stage does not have. It should be renamed to an explicit `PLACEHOLDER_DA_DT` constant with a TODO referencing the Stage 3 temporal module, so it cannot silently be read as a measured value.

---

## What this taught me

The most valuable output of a dataset search wasn't a dataset. It was establishing that a particular intersection is empty, and finding published evidence that the gap causes total failure rather than mild degradation. That converts "I annotated my own data" from a fallback into a justified work package — and the justification is stronger for having been reached by actually searching rather than assumed.

Also worth noting how the FLAME 2 problem surfaced: not from reading the documentation, but from noticing that a 543-byte file cannot possibly hold 53,541 sets of box coordinates. Order-of-magnitude checks catch things that descriptions gloss over.

---

## Contribution log

**My reasoning:** questioned whether the Roboflow results were actually suitable rather than treating the result count as a measure of available choice; directed the evaluation sequence (FLAME 2 scope, FLAME 3 archive selection, Roboflow filtering) and asked for the filtering to be performed rather than described; identified that the FLAME 3 readme belonged to the NADIR plot archive, which corrected an earlier mis-assessment; asked whether `da_dt=5.0` was computed or assumed, surfacing a placeholder that would otherwise read as a real output; requested a second consolidation pass over the data sources rather than treating the first survey as final, which is what surfaced FASDD.

**AI assistance:** executed the dataset surveys; located the Boreal Forest Fire dataset and its paper; detected the 737-image duplication pattern and verified the canonical source; extracted the annotation-strategy and generalisation-failure figures; drafted this record.

**Open for my own follow-up:**
- Verify Kapustin annotation quality directly before including it — currently unverified.
- Determine how many self-annotated local frames are actually needed. The Boreal result suggests a lightweight detector can reach usable accuracy on relatively little high-quality data, but the number for this project has to come from my own validation curves, not from citing theirs.
- Confirm whether FLAME 2 #9's pre-registration is close enough to treat as Identity, or whether a residual offset needs measuring.
- Validate the large-box strategy on a small local sample before committing to it wholesale.

---

# Amendment A — FASDD found; the "empty intersection" claim narrowed

**Date:** 2026-07-29 (same day, second consolidation pass)

The record above is left unedited. This amendment states what it got wrong and what
replaces it.

## What prompted it

Rather than treating the first survey as final, I asked for a second consolidation
pass over training-data sources before committing to the plan. That pass found
**FASDD** (Flame And Smoke Detection Dataset), which the first survey missed
entirely.

## What FASDD changes

FASDD is a 100,000-level detection dataset with flame and smoke labelled as `fire`
and `smoke`, published in VOC, YOLO, COCO and TDML formats. It splits into three
sub-datasets by sensor platform: **FASDD_CV** (ground-based), **FASDD_UAV**
(airborne), **FASDD_RS** (spaceborne). Reported Swin Transformer performance is
mAP 84.9% / 89.7% / 74.0% respectively.

Two claims above are therefore wrong:

| Original claim | Corrected |
|---|---|
| No public source provides **flame** boxes; flame must come from converted segmentation masks or direct annotation | FASDD provides both `fire` and `smoke` boxes in YOLO format |
| The only UAV-perspective detection dataset is Boreal, which labels smoke only | FASDD_UAV is a dedicated airborne sub-dataset carrying both classes |

## What survives

The central finding is **narrowed, not overturned**. FASDD's imagery is drawn from
mixed sources — existing open datasets, social media, CG renders, UAV imagery, web
crawls — not from the target deployment region.

```
Original:   (UAV perspective × local vegetation) = EMPTY
            → both halves unmet

Corrected:  UAV perspective   — now covered by FASDD_UAV
            local vegetation  — still unmet
            (UAV × local)     — still EMPTY
```

Self-annotation of local aerial frames remains a required work package, but **the
justification narrows** from "no UAV detection data exists" to "no data from this
background exists." The Ruokolahti generalisation evidence (0.93 → 0.031) supports
the narrower claim directly, since that failure was specifically a background shift,
not a viewpoint shift.

FASDD corroborates the same effect internally: reviewer discussion of the dataset
paper notes that FASDD_CV knowledge transfers poorly to FASDD_RS, with combined
training *degrading* FASDD_RS performance. **Practical consequence: train on
FASDD_UAV alone first and establish a baseline before mixing sub-datasets.** Merging
CV + UAV + RS at the outset may hurt the airborne case.

## Additional source: D-Fire

Also found in the second pass. 21,527 images, 26,557 boxes (11,865 `smoke`, 14,692
`fire`), YOLO format, 416×416. Composition: 1,164 fire-only, 5,867 smoke-only, 4,658
both, and **9,838 `none`** — described by the authors as images containing objects or
environments that could be mistaken for fire or smoke.

That `none` set is a ready-made **hard-negative** pool for Decision 002, which would
otherwise have required collecting fog and foliage imagery by hand.

**Quality caveat, from an independent dataset review:** D-Fire contains substantial
duplicate imagery, a large proportion of background images, predominantly dark
backgrounds behind fire and smoke, and synthetic smoke composited onto green
landscape backgrounds. Suitable for **volume and negatives, not as a primary training
source** — the dark-background bias in particular is the opposite of a daytime patrol
scenario.

## Revised source priority

| Priority | Source | Scale | Classes | Role |
|---|---|---|---|---|
| 1 | FASDD_UAV | 25,097 samples | fire + smoke | Primary — airborne, both classes, YOLO-ready |
| 2 | FASDD_CV | 95,314 samples | fire + smoke | Volume; mix only after a UAV-only baseline exists |
| 3 | Boreal Forest Fire | 4,954 | smoke | High-quality UAV smoke, peer-reviewed annotation |
| 4 | D-Fire | 21,527 | fire + smoke | Volume + 9,838 hard negatives |
| 5 | HPWREN / AI for Mankind | 737 | smoke | Local geography, ground perspective |
| 6 | Self-annotated local frames | TBD | fire + smoke | Local background adaptation — still required |

**Negative samples:** FASDD `NeitherFireNorSmoke` (the category is encoded as a
filename prefix, so filtering is trivial) + D-Fire's 9,838 `none` + Boreal's 256
empty-label images. Split into general and hard subsets per Decision 002.

## Download links

```
FASDD              https://doi.org/10.57760/sciencedb.j00104.00103
  paper (ESSD)     https://essd.copernicus.org/preprints/essd-2023-73/
  Kaggle mirror    https://www.kaggle.com/datasets/yuulind/fasdd-cv-coco
                   (CV subset, COCO format only)

D-Fire             https://github.com/gaiasd/DFireDataset

Boreal Forest Fire https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9
  paper            https://www.nature.com/articles/s41597-025-05634-0

HPWREN             https://github.com/aiformankind/wildfire-smoke-dataset
  Roboflow mirror  universe.roboflow.com/brad-dwyer/wildfire-smoke
```

## FASDD — verified from the Science Data Bank record

Checked directly against the dataset landing page (V9, updated 2025-09-11). The two
items previously flagged as unverified are now resolved.

**Licence: CC BY-SA 4.0** — Attribution + ShareAlike. Commercial use is **permitted**;
there is no NonCommercial term. This is less restrictive than HPWREN's BY-NC-SA, so
the commercial-design-target concern recorded earlier does not apply to this source.

The ShareAlike term is the open question: whether a model trained on BY-SA data
constitutes a derivative work that must itself be released under BY-SA is legally
unsettled and I am not qualified to resolve it. Irrelevant for a personal or academic
project; needs proper advice before any commercial use.

**Citation is a stated requirement of use:**

> Wang, M., Yue, P., Jiang, L., Yu, D., Tuo, T., & Li, J. (2025). An open flame and
> smoke detection dataset for deep learning in remote sensing based fire detection.
> *Geo-spatial Information Science*, 28(2), 511–526.

**Composition (V9):**

| | Samples | Fire instances | Smoke instances |
|---|---|---|---|
| FASDD_CV | 95,314 | 73,297 | 53,080 |
| **FASDD_UAV** | **25,097** | **36,308** | **17,222** |
| FASDD_RS | 2,223 | 3,549 | 2,770 |
| **Total** | **122,634** | 113,154 | 73,072 |

Of the 122,634 total, 70,581 are positive samples and **52,073 are negatives** —
consistent with the `NeitherFireNorSmoke` filename prefix and confirming the negative
pool is large enough to source both general and hard negatives per Decision 002.

**Distribution:** four archives — `FASDD_CV.zip`, `FASDD_UAV.zip`, `FASDD_RS.zip`,
and `FASDD_RS_SWIR.zip` (pseudo-colour SWIR imagery for flame detection in remote
sensing). Each contains `images/` and `annotations/`, with annotations in YOLO, VOC,
COCO and TDML, pre-split 1/2 : 1/3 : 1/6 into train/val/test. **Only `FASDD_UAV.zip`
is needed** for the minimal-integration plan. Total published volume is 76.49 GB
across 6 files; the per-archive breakdown is not given on the landing page.

**Record the version used.** The dataset is at V9 and has been revised nine times
since 2022. Secondary sources describe earlier compositions — one cites FASDD_RS at
5,773 samples against the current 2,223 — so any figure taken from a paper or mirror
rather than the landing page may describe a different version. Note V9 in the
training config.

## Class imbalance in FASDD_UAV — relevant to this project specifically

The UAV subset carries **36,308 fire instances against 17,222 smoke instances**, a
ratio of roughly 2:1 in favour of fire.

This runs against the project's priority. Smoke appears earlier and is visible from
further away than open flame, which is the basis of the early-warning claim — so
smoke is the more valuable class here, and it is the minority class in the primary
training source.

Two consequences:

- **Report per-class AP, not just overall mAP.** An overall figure will be carried by
  the majority class. If smoke AP trails fire AP, that is a data-distribution effect
  rather than an algorithmic failure, and should be described as such.
- This is a concrete entry point for the deferred YOLOv8 work: class weighting,
  resampling, or supplementing with Boreal's smoke-only 4,954 images are all
  responses to a problem that is now measured rather than assumed.

## Possibly relevant to Phase 2 — noted, not verified

The Science Data Bank record lists a related dataset: **"Non-Spatially Aligned LLVIP,
M3FD, and FLIR Datasets"**, tagged non-spatial registration / RGB-IR / object
detection / multimodal. Deliberately *un*registered RGB-IR pairs would be a more
honest test input for registration work than aligned pairs with synthetic offsets
applied. Not examined; recorded here so it is not lost.

## What this taught me

The first survey felt thorough enough to be conclusive and was still wrong on two
counts. What corrected it was not new reasoning but running the search again with a
different framing — "what trains flame and smoke" rather than "what wildfire datasets
exist." A negative finding ("this doesn't exist") is a much weaker claim than it
feels like, because it is bounded by the search that produced it, not by reality.

Worth carrying into the geolocation and calibration work: a conclusion of the form
"there is no X" should travel with the search that justified it, so a future reader
knows what would overturn it.

## Contribution log — Amendment A

**My reasoning:** called for a second pass instead of accepting the first survey as
settled. That request is the entire reason this correction exists.

**AI assistance:** ran the second survey; located and verified FASDD and D-Fire;
extracted the sub-dataset structure, class scheme, and quality caveats; drafted this
amendment.

**Open for my own follow-up:**
- Train a FASDD_UAV-only baseline before any sub-dataset mixing, and record whether
  mixing helps or hurts. The reviewer critique predicts it may hurt; that prediction
  is worth testing rather than assuming.
- Watch smoke AP specifically against fire AP, given the 2:1 imbalance.
- Check the non-aligned LLVIP/M3FD/FLIR dataset for Phase 2 registration validation.

*(FASDD size and licence, previously listed here, were resolved by direct check of
the Science Data Bank record — see the verification section above.)*
