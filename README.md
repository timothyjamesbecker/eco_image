# eco_image
Tools for automatic manipulation of ecological image data sets

### (1) exif metadata retrieval using the pypy exifread
```python
import exifread
with open(path,'rb') as f: tags = exifread.process_file(f)
```
