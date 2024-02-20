# Usage

Use the wrapper in your code :

```python
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper

MSW = MobileSamWrapper()
masks = MSW.infer(<image>, [bbox1, bbox2 ...])
```