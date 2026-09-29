import os

data_dir = 'archive/images'
removed = 0

for root, _, files in os.walk(data_dir):
    for fname in files:
        path = os.path.join(root, fname)
        with open(path, 'rb') as f:
            is_jfif = b'JFIF' in f.peek(10)   # real JPEGs carry this marker near the start
        if not is_jfif:
            os.remove(path)
            removed += 1

print(f'removed {removed} files')