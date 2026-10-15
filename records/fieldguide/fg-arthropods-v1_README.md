# fg-arthropods-v1

Fieldguide photos of arthropods, mollusks and worms with one box per photo, for
training object detectors. Built from earlier Fieldguide detector sets without
downloading any image again. Version v1.0.0, built 2026-10-12T19:39:12Z
from commit `aba9a998b`.

20,642 photos, 22.3 GB of images.

## Layout

- `index.parquet` and `index.jsonl`: one row per photo, same columns in both.
- `images/<sha256[:2]>/<sha256>.jpg`: each image is named by the SHA-256 of its bytes.
- `SHA256SUMS`: checksum of every image (`sha256sum -c SHA256SUMS`).
- `METADATA.json`: summary statistics, overall and per group: box area as a share of
  the image (histogram and buckets), box and image sizes in pixels, licences, and the
  most common orders and families (`scripts/dataset_metadata.py`).
- `taxa.parquet`: every Fieldguide category the rows refer to, with its rank,
  parent and lineage names.
- `attribution.csv`: photo id, licence text and licence class for every row.
- `spec.json`: sources, counts, rules and the duplicates that were dropped.
- `private/`: photographer ids. Do not copy this folder outside the project.

## Columns

`dataset_name`, `dataset_version`, `source_set`, `source_dataset_version`, `data_source`, `extracted_at`, `extract_commit_sha`, `image_key`, `sha256`, `image_size_bytes`, `photo_id`, `image_pk`, `image_id`, `width`, `height`, `bbox_xyxy`, `bbox_area_fraction`, `bbox_is_square`, `box_kind`, `bucket`, `split`, `category_id`, `parents`, `group`, `order`, `family`, `taxon_name`, `rank`, `license`, `license_class`, `created_at`

- `bbox_xyxy` is in pixels of the stored image. The box is the square crop the
  user made at upload (`box_kind = "user_crop"`), so it can be loose.
- `group` is one of `butterfly`, `moth` (Lepidoptera that are not butterflies),
  `other_arthropod` and `mollusk_worm`. `order`, `family`, `taxon_name` and `rank`
  come from Fieldguide's taxonomy; `order` is null where Fieldguide has no
  order-level node (some snails and slugs, unplaced insects).
- `bucket` is the box area as a share of the image: small < 0.10 <= medsmall
  < 0.25 <= medlarge < 0.50 <= large < 0.75 <= full.
- `source_set` names the earlier set a row came from.

## Splits

Every row keeps the train/val split it had in its source set, so scores on the
earlier val sets stay comparable. Photos added later use one rule:
`md5("fg-arthropods-v1:" + photo_id) < 0.1 -> val`.

| Group | Rows |
|---|---|
| butterfly | 3,518 |
| moth | 15,796 |
| other_arthropod | 1,017 |
| mollusk_worm | 311 |

| Bucket | Rows |
|---|---|
| small | 6,058 |
| medsmall | 14,580 |
| medlarge | 4 |

| Source set | Rows | Kept | Duplicates dropped | Image missing |
|---|---|---|---|---|
| butterflies-fg-medsmall-4200 | 3,520 | 3,518 | 2 | 0 |
| invertebrates-fg-moths-small | 5,847 | 5,822 | 25 | 0 |
| invertebrates-fg-moths-medsmall-10000 | 9,986 | 9,974 | 12 | 0 |
| invertebrates-fg-other-smallmed | 1,332 | 1,328 | 4 | 0 |

## Licences

The licence is Fieldguide's free-text copyright field. It mixes Creative Commons
licences, "all rights reserved" and bare photographer names, and it was not
recorded for every source set (`license_class = "missing"`). Check
`license_class` before using a row outside this project; only `cc0`, `cc-by` and
`cc-by-sa` allow redistribution without further permission.

| Licence class | Rows |
|---|---|
| all-rights-reserved | 1,332 |
| cc-by | 241 |
| cc-by-nc | 3,352 |
| cc-by-nc-nd | 99 |
| cc-by-nc-sa | 82 |
| cc-by-nd | 3 |
| cc-by-sa | 229 |
| cc0 | 24 |
| missing | 3,523 |
| unspecified | 11,757 |
