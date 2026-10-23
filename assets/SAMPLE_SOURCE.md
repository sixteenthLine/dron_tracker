# Sample Media Source

## Published Aerial Images

`aerial/park.jpg` and `aerial/terrain.jpg` are two full-frame examples from the
local **HERIDAL v2 yolo_non_412 Roboflow export**. They depict civilian outdoor
scenes from above, consistent with the dataset's search-and-rescue research scope.

- Credit: **Dunja Bozic-Stulic, Zeljko Marusic, Sven Gotovac / HERIDAL dataset**;
  Roboflow export. Preview assembly: Aerial Object Detection contributors.
- [Original dataset page](http://ipsar.fesb.unist.hr/HERIDAL%20database.html).
- [Current public export page](https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z).
- Export license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  declared in both the local COCO image-license records and export README.
- Upstream [AIR dataset distributor](https://github.com/Accenture/AIR#licenses)
  documents original HERIDAL data as [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
  These source declarations are recorded separately; neither is replaced by MIT.
- Export date: October 17, 2022. Samples prepared: October 22, 2026.
- Changes: full images resized from 4000 x 3000 to 1600 x 1200 and JPEG encoded
  at quality 90; no cropping or enhancement; metadata is not copied.
- Exact source filenames, image IDs, dimensions, and before/after SHA-256 hashes
  are in [aerial/PROVENANCE.json](aerial/PROVENANCE.json).

`docs/media/aerial_samples.jpg`, `detection.jpg`, and `detection_2.jpg` are
resized comparison layouts of these same samples. They retain the same source
credits and export licensing. The output panels show actual pretrained inference:
**zero boxes at threshold 0.50 on each image**. No reference annotations are
drawn as predictions. No endorsement by dataset authors is implied.

## Reference Annotation Previews

`aerial/reference_annotations.coco.json` contains only the 16 original boxes for
the two published images: 15 for park, one for terrain. Coordinates are scaled
from the original 4000 x 3000 frames to the included 1600 x 1200 JPEGs. No boxes
are invented or manually adjusted, and no model confidence scores are attached.
The source annotation hash and transformation are recorded in that COCO file;
its own SHA-256 is recorded in `aerial/PROVENANCE.json`.

`docs/media/reference_park.jpg` and `reference_terrain.jpg` visualize those
reference labels. Both carry **REFERENCE ANNOTATIONS / NOT MODEL OUTPUT** in the
image itself. They are layout/data examples, not detector results or metrics.
`docs/media/REFERENCE_PREVIEWS.json` records source hashes, box counts, output
dimensions, and hashes. The images and annotation subset retain the dataset's
export license declaration and original credits, not the source-code MIT license.

The full dataset stays local and ignored. These two samples are used only for
offline inference/preview examples, not training, tracking, or benchmark evaluation.
Their presence does not establish the identity of historical training data.

## Previous Local Images

`sample_public_dog.jpg` (Ruerd Kroon, Wikimedia Commons, CC BY-SA 4.0) is no
longer part of the published demo and is ignored by Git. Its previous preview
has been replaced by aerial imagery.

`sample_civilian_dog.jpg` came from
<https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg> during
Level A. Its photo license was not established. It remains on the local machine
but is ignored by Git and is not used by current public demo instructions.
