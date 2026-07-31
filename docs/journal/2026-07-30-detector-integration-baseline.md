# 2026-07-30 — Detector integration baseline on FASDD_UAV

## What I set out to do

Train a minimal flame/smoke detector — good enough to replace the simulated detection
boxes the fusion stage has been running on, so the pipeline can be exercised end to end
with real inputs. Detector quality was explicitly not the objective; getting real boxes
flowing through the fusion stage was.

Single data source: the UAV sub-dataset of FASDD. It ships both classes in YOLO format
and includes its own negative images, so it covers the requirements on its own without
mixing sources.

## What I actually did

YOLOv8n, first 10 backbone layers frozen, COCO-pretrained initialisation, 640 px input,
3 epochs on CPU. Evaluated on FASDD_UAV's own test split.

```
overall mAP@50        0.7367
overall mAP@50-95     0.4757

per-class AP@50
  flame               0.6208
  smoke               0.8526

negatives (1,997 images, conf 0.25)
  false positive boxes        33
  false positives per image   0.0165
```

## What surprised me

I expected smoke to be the weaker class and it is the stronger one, by 23 points.

The prediction came from the instance counts: FASDD_UAV contains 36,308 flame instances
against 17,222 smoke instances, roughly 2:1. Smoke is the minority class, and it is also
the class this project cares about more — smoke is visible earlier and from further away
than open flame, which is the whole basis for treating this as an early-warning problem.
So I asked for per-class AP specifically to see how badly smoke was lagging.

It is ahead instead. Two hypotheses I have not tested:

- Flame in aerial imagery is often small and its boundary shifts with exposure, so the
  boxes are harder to fit consistently.
- Smoke plumes are large and continuous, which suits the large-box annotation strategy
  this dataset follows. Bigger targets are easier to localise at IoU 0.5.

Both are guesses. What I can say is that the instance-count argument I reasoned from does
not predict the outcome here, so class frequency is not the dominant factor — something
about the geometry of the targets matters more.

The thing I want to keep from this: **overall mAP would have hidden it entirely.** 0.7367
sits between the two class values and looks unremarkable. The reason I saw anything at all
is that I had asked for the classes to be reported separately, and I asked for that in
order to confirm a weakness that turned out not to exist. The request was worth making and
the reasoning behind it was wrong — those are separate things.

## What I am not concluding

3 epochs on CPU, one run, no repeats. This is an observation about one training run, not a
measurement of the two classes' relative difficulty. A longer run could reorder them.

The false-positive rate also needs reading alongside recall before it means much. A
model trained this briefly may simply be producing few detections overall, which would
deflate the false-positive count for reasons that have nothing to do with the negatives
being handled well. Worth checking directly rather than taking 0.0165 at face value.

## Next

- Wire the trained weights into the fusion stage and retire the simulated boxes. That was
  the point of this run.
- Check the false-positive figure against detection counts before treating it as a
  meaningful number.
- The growth rate feeding the fusion score is still a fixed constant rather than a tracked
  value. That is the other placeholder, and it needs a multi-frame source before it can be
  replaced.
