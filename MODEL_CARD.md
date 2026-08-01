# Model card

`models/shape_vectorizer_v2.pt` is the bundled 450-slot initializer used by the UI and CLI.

## Training data

The model was first warmed up on generated vector scenes, then trained on random crops from
DIV2K training images `0001.png` through `0799.png`. No raw training images are included in this
repository. DIV2K is distributed separately by ETH Zurich for academic research; review the
[official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) before downloading or using it.

## Intended use

The checkpoint is intended for research, experimentation, and local image-to-SVG conversion. It
predicts an initial composition of up to 450 ellipses, rectangles, and triangles. Detailed mode
then optimizes those objects against the supplied image.

## Limitations

The representation is deliberately compact. It produces a stylized approximation rather than a
pixel-perfect trace, and its results depend on the subject and requested object count. The model
does not determine whether an input image may legally be copied or transformed; users are
responsible for images they supply.

The source code is covered by the repository's MIT license. Dataset terms and third-party image
licenses remain separate.
