# Domain Data Loading

## Table of Contents

1. [Image Data Loading](#1-image-data-loading)
2. [Text Data Loading](#2-text-data-loading)
3. [Tabular and CSV Data](#3-tabular-and-csv-data)
4. [Audio and Video](#4-audio-and-video)
5. [TFRecord and Custom Formats](#5-tfrecord-and-custom-formats)
6. [From Directories and File Patterns](#6-from-directories-and-file-patterns)
7. [Domain-Specific Datasets](#7-domain-specific-datasets)
8. [Best Practices](#8-best-practices)

---

## 1. Image Data Loading

### Image Files from Directory

**tf.keras.utils.image_dataset_from_directory** creates a labeled dataset from a directory structure:

```
data/
  train/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      img1.jpg
  val/
    class_a/
    class_b/
```

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42
)
```

**Key parameters:**
- **image_size**: Resize all images to this (height, width)
- **batch_size**: Samples per batch
- **validation_split**: Fraction for validation
- **subset**: 'training' or 'validation'
- **label_mode**: 'int', 'categorical', 'binary'

### Decoding Images Manually

```python
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return img

ds = tf.data.Dataset.list_files('images/*.jpg')
ds = ds.map(lambda p: load_image(p))
```

---

## 2. Text Data Loading

### Text Files

```python
ds = tf.data.TextLineDataset('file.txt')
ds = ds.map(lambda line: tf.strings.split(line, sep=','))
```

### From Directory

```python
files = tf.io.gfile.glob('texts/*.txt')
ds = tf.data.Dataset.from_tensor_slices(files)
ds = ds.map(lambda f: tf.io.read_file(f))
```

### Tokenization Pipeline

```python
tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=128)
tokenizer.adapt(ds.map(lambda x, y: x))
ds = ds.map(lambda x, y: (tokenizer(x), y))
```

---

## 3. Tabular and CSV Data

### tf.data.experimental.CsvDataset

```python
ds = tf.data.experimental.CsvDataset(
    'data.csv',
    record_defaults=[tf.float32, tf.float32, tf.int32],
    header=True
)
```

### Pandas to Dataset

```python
import pandas as pd
df = pd.read_csv('data.csv')
ds = tf.data.Dataset.from_tensor_slices((df[features].values, df['label'].values))
```

### CSV with tf.io

```python
def parse_csv(line):
    fields = tf.io.decode_csv(line, record_defaults=[...])
    return fields[:-1], fields[-1]

ds = tf.data.TextLineDataset('data.csv').skip(1).map(parse_csv)
```

---

## 4. Audio and Video

### Audio (WAV)

```python
def load_audio(path):
    audio_binary = tf.io.read_file(path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    return audio, sample_rate

ds = tf.data.Dataset.list_files('audio/*.wav').map(load_audio)
```

### Video (Frames)

Use `tf.io.read_file` and decode with appropriate codec, or use `tf.data` with frame extraction libraries. TensorFlow I/O provides additional format support.

---

## 5. TFRecord and Custom Formats

### TFRecord

**TFRecord** is TensorFlow's binary format for efficient storage and loading.

```python
def parse_tfrecord(example):
    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(example, feature_desc)
    image = tf.io.decode_image(parsed['image'])
    return image, parsed['label']

ds = tf.data.TFRecordDataset('data.tfrecord')
ds = ds.map(parse_tfrecord)
```

### Writing TFRecord

```python
def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def write_tfrecord(images, labels, path):
    with tf.io.TFRecordWriter(path) as writer:
        for img, lbl in zip(images, labels):
            writer.write(serialize_example(img, lbl))
```

---

## 6. From Directories and File Patterns

### list_files

```python
files = tf.data.Dataset.list_files('images/*.jpg')
files = files.shuffle(1000)
```

### Glob

```python
import glob
files = glob.glob('data/**/*.png')
ds = tf.data.Dataset.from_tensor_slices(files)
```

---

## 7. Domain-Specific Datasets

### TensorFlow Datasets (tfds)

```python
import tensorflow_datasets as tfds
ds, info = tfds.load('mnist', split='train', with_info=True)
```

### Keras Datasets

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
```

| Source | Method |
|--------|--------|
| Images | image_dataset_from_directory |
| Text | TextLineDataset |
| CSV | CsvDataset, from_tensor_slices |
| TFRecord | TFRecordDataset |
| Audio | decode_wav, tf.io.read_file |

---

## 8. Best Practices

| Practice | Description |
|----------|-------------|
| Use TFRecord for large datasets | Efficient I/O, compression |
| Decode in map for images | Parallel decode with num_parallel_calls |
| Match batch size to GPU memory | Avoid OOM |
| Shuffle file list before loading | Avoid order bias |
