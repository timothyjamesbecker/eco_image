import exifread

# Given an image file with exif metadata, return set of tags that are required
# as well as a checksum and the hex of the file.
def get_exif_tags(path, tag_set=[]):

    tags, T = {}, {}
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
    if not tag_set:
        for t in tags:
            if t not in ('JPEGThumbnail', 'TIFFThumbnail'):
                T[t] = str(tags[t]).rstrip(' ')
    else:
        for t in tag_set:
            if tags.get(t) is not None:
                T[t] = str(tags[t]).rstrip(' ')
            else:
                T[t] = None
    T["Path"] = path
    return T