# Usage

Use the wrapper in your code :

```python
from vision_models.mobile_sam.mobile_sam_wrapper import MobileSamWrapper

MSW = MobileSamWrapper()
masks = MSW.infer(<image>, [bbox1, bbox2 ...])
```