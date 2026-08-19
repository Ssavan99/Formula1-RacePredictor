# 3D model credit

**McLaren MP4/5 || Formula 1** by [dark_igorek](https://sketchfab.com/dark_igorek),
licensed [CC Attribution](https://creativecommons.org/licenses/by/4.0/), via
[Sketchfab](https://sketchfab.com/3d-models/mclaren-mp45-formula-1-3059d4532ecd48ca8da41e1cac971f22).

Processed for the web with `@gltf-transform/cli`: Draco geometry compression,
textures re-encoded to WebP at 1024px, unused UV sets pruned. **105.8 MB -> 2.5 MB.**
Mesh joining was disabled so all 16 parts stay separable for the exploded view.
Geometry is otherwise unmodified.

Reproduce with:

    npx @gltf-transform/cli@3.10.1 optimize <source>.glb docs/model/mp45.glb \
      --compress draco --texture-compress webp --texture-size 1024 \
      --simplify false --join false
