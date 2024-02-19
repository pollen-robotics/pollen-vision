# Usage

Use the wrapper in your code :

```python
from vision_models.owl_vit.owl_vit_wrapper import OwlVitWrapper

OW = OwlVitWrapper()
predictions = OW.infer(<image>, ["object1", "object2", ...])

bboxes = OW.get_bboxes(predictions)
labels = OW.get_labels(predictions)
```