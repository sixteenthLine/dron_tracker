# Third-Party Notices

The MIT license in this repository covers new reconstruction code and original
documentation. It does not relicense photographs, datasets, pretrained weights,
or third-party libraries.

## Included Aerial Samples And Previews

- Files: `assets/aerial/park.jpg`, `assets/aerial/terrain.jpg`,
  `docs/media/aerial_samples.jpg`, `docs/media/detection.jpg`, `docs/media/detection_2.jpg`.
  Also: `assets/aerial/reference_annotations.coco.json`,
  `docs/media/reference_park.jpg`, and `docs/media/reference_terrain.jpg`.
- Credit: **Dunja Bozic-Stulic, Zeljko Marusic, Sven Gotovac / HERIDAL dataset**;
  Roboflow export. Preview assembly by Aerial Object Detection contributors.
- Export: <https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z>.
- Export metadata and the public page declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The [AIR distributor's license notice](https://github.com/Accenture/AIR#licenses)
  describes original HERIDAL data as [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
  Both declarations and original credits are retained; MIT does not cover images.
- Changes: resized full frames, JPEG encoding, metadata omitted, comparison
  layouts, and a separate reference-annotation overlay. Reference boxes are
  scaled source annotations, explicitly not predictions. No cropping,
  fabricated predictions, or author endorsement.
- Exact provenance: [assets/SAMPLE_SOURCE.md](assets/SAMPLE_SOURCE.md) and
  [sample manifest](assets/aerial/PROVENANCE.json).

## Synthetic Preview

`docs/media/tracking.gif` is generated from this project's synthetic frame source,
mock detector, IoU tracker, and visualization code. It contains no external
footage or historical dataset images. It is included under the project MIT license.

## Dependencies And Model Weights

Dependencies are installed from their upstream distributions; their source code
is not vendored here. Each retains its upstream license:

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE).
- [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE).
- [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE).
- [ImageIO](https://github.com/imageio/imageio/blob/master/LICENSE).
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE).
- [pycocotools](https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/LICENSE),
  optional COCO evaluation dependency.

The image demo requests torchvision's public
`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT` weights. Weights are
downloaded into the user's Torch cache and are not redistributed here. They are
not historical project weights, and this project's MIT license does not cover them.
New fine-tuned checkpoints derive from those public weights, remain local under
`outputs/training/`, and are not included in the source bundle or relicensed as MIT.

## Excluded Local Materials

- `Heridal.v2-yolo_non_412.coco/`: local export metadata declares **CC BY 4.0**
  and links to <https://universe.roboflow.com/object-detection/heridal-t428z>.
  The full export is not redistributed. Only the two attributed sample frames
  above and their 16-box reference annotation subset are included; all other
  annotations/images stay local. Historical identity remains unverified.
- `assets/sample_civilian_dog.jpg`: previous PyTorch Hub sample; redistribution
  license not established. Kept locally and ignored, as are old derived outputs.
- `assets/sample_public_dog.jpg`: former CC BY-SA 4.0 demo photo, now also local
  and ignored; the published previews use the aerial samples instead.
- Historical GOTURN binaries and archived source: not redistributed or imported.

See [dataset status](docs/DATASET_STATUS.md) for the distinction between local
files, historical audit evidence, and publication contents.
