# Usage

Use the wrapper in your code :

```python
from pollen_vision.vision_models.object_detection import OwlVitWrapper

OW = OwlVitWrapper()
predictions = OW.infer(<image>, ["object1", "object2", ...])

bboxes = OW.get_bboxes(predictions)
labels = OW.get_labels(predictions)
```