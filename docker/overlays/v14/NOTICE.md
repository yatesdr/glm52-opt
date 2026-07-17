# v1.4 serving overlays

This directory vendors exactly the seven v1.4 overlay files present in the
verified GLM-5.2 production container. They originate from the Apache-2.0
serving-stack work by David Young (`davidsyoung/vllm-glm52`) and the vLLM
project. They are redistributed under this repository's Apache-2.0 `LICENSE`.

The directory intentionally excludes seven other files found in the source
overlay bundle: three were not mounted in production and four are superseded
by the stage-3 or phase-2 overlays in this repository. `docker/overlay-md5.txt`
pins every file installed in the final image.
