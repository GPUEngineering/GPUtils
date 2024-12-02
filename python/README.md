## GPUtils API

### Installation

As simple as...
```bash
pip install gputils-api
```
of course, preferably from within a virtual environment.

### Write to file

```python
 import numpy as np
import gputils_api as g
a = np.eye(3)
g.write_array_to_gputils_binary_file(a, 'my_data.bt')
```

### Read from file

```python
import numpy as np
import gputils_api as g
x = g.read_array_from_gputils_binary_file('my_data.bt')
```


