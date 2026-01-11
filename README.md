# Intrinsic Image Decomposition

---

| Key          | Description |
| --------     | -------     |
| ~'image'~      | input image after specified resizing scheme   |
| ~'lin_img'~    | input image after undoing gamma correction |
| ~'ord_full'~   | full-resolution ordinal shading estimation |
| ~'ord_full'~   | base-resolution ordinal shading estimation  |
| ~'gry_shd'~    | grayscale shading from ordinal pipeline     |
| ~'gry_alb'~    | implied albedo from the ordinal pipeline (img / gry_shd)   |
| ~'lr_clr'~     | estimated low-resolution shading chromaticity |
| ~'lr_alb'~     | implied albedo after shading chromaticity estimation (img / lr_shd) |
| ~'lr_shd'~     | high-resolution grayscale shading + low-res chroma   |
| '**hr_alb**'     | **final high-resolution albedo estimation**    |
| ~'hr_shd'~     | implied shading of the final albedo (img / hr_alb) |
| ~'hr_clr'~     | visualized chroma of hr_shd   |
| ~'wb_img'~     | white-balanced image - hr_alb * luminance(hr_shd)    |
| ~'dif_shd'~    | diffuse shading estimation   |
| ~'dif_img'~    | diffuse image (hr_alb * dif_shd)    |
| ~'residual'~  | residual component (img - dif_img)    |
| ~'neg_res'~    | negative part of the residual component (due to saturated input pixels)    |
| ~'pos_res'~    | positive part of the residual component (specularities, light sources, etc)  |
