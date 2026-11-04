# EDA robustness checks

Creator cap: at most 3 videos per creator and platform, chosen at random (seed 42). This is a sensitivity analysis, not a correction of the data.

## Creator cap: strict dataset

Kept 160 of 160 videos (100.0%).

- all videos: 160 videos, YouTube 59.4%, median duration 124 s, median views 3,918, median like rate 0.95%, transcript 18.1%; top settings: imaging (MRI, X-ray, CT) 22%; needle procedures 18%; not specified in text 17%.
- cap 3 per creator: 160 videos, YouTube 59.4%, median duration 124 s, median views 3,918, median like rate 0.95%, transcript 18.1%; top settings: imaging (MRI, X-ray, CT) 22%; needle procedures 18%; not specified in text 17%.
- YouTube vs TikTok duration, Cliff's delta: +0.88 (all videos) vs +0.88 (capped); p < 0.001 vs p < 0.001.

## Strict set: human-labeled vs model-selected

- human-labeled: 104 videos, median duration 120 s, median views 5,372, transcript 23.1%, non-English text 10.6%.
- model-selected: 56 videos, median duration 141 s, median views 2,736, transcript 8.9%, non-English text 12.5%.
- Duration (human vs model): median 120 vs 141 s, Cliff's delta -0.01 (negligible), p = 0.920.

If the two sources differ a lot (for example in transcript share), the model-selected part of the strict set is a different kind of video from the human-labeled part, and conclusions about the strict set should say so.

## Creator cap: all dataset

Kept 2,195 of 2,319 videos (94.7%).

- all videos: 2,319 videos, YouTube 48.5%, median duration 92 s, median views 15,693, median like rate 1.35%, transcript 26.3%, relevant 6.9%; top settings: not specified in text 28%; trauma / spinal / splint 15%; needle procedures 14%.
- cap 3 per creator: 2,195 videos, YouTube 48.4%, median duration 90 s, median views 14,192, median like rate 1.37%, transcript 25.9%, relevant 7.1%; top settings: not specified in text 29%; trauma / spinal / splint 15%; needle procedures 13%.
- YouTube vs TikTok duration, Cliff's delta: +0.79 (all videos) vs +0.79 (capped); p < 0.001 vs p < 0.001.

## Reading these results

If platform shares, medians and top settings stay close after the creator cap, the main EDA findings do not depend on a few prolific creators. Search-result sampling can still favour certain creators, platforms and kinds of video.

## Tables

- `tables/01_creator_cap_strict.csv`
- `tables/02_human_vs_model_strict.csv`
- `tables/01_creator_cap_all.csv`
