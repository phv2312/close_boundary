## Usage

```python
import cv2
from algo.rule1.closing import ClosingModel

closing_model = ClosingModel(max_traveled_pixel=5, max_pair_distance=25, keypoint_to_boundary_distance=20)

sketch_tgt_fn = "<where is your sketch image path>"
pair_points, _, _, _ = closing_model.process(np.array(Image.open(sketch_tgt_fn).convert('L')))

import cv2
im = np.asarray(Image.open(sketch_tgt_fn))
for p1, p2 in pair_points:
    cv2.line(im, (p1[1], p1[0]), (p2[1], p2[0]), color=(0,0,0), thickness=1)

cv2.imwrite('output.png', im)
```
