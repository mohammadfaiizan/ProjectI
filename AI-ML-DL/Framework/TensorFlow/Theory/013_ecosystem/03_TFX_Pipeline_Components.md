# TFX Pipeline Components

## Table of Contents

1. [TFX Overview](#1-tfx-overview)
2. [ExampleGen](#2-examplegen)
3. [StatisticsGen and SchemaGen](#3-statisticsgen-and-schemagen)
4. [ExampleValidator](#4-examplevalidator)
5. [Transform](#5-transform)
6. [Trainer](#6-trainer)
7. [Evaluator and Pusher](#7-evaluator-and-pusher)
8. [Pipeline Orchestration](#8-pipeline-orchestration)

---

## 1. TFX Overview

**TensorFlow Extended (TFX)** is a platform for production ML pipelines. Pipelines are composed of **components** that produce **artifacts** consumed by downstream components.

**Key concepts:**
- **Component:** A unit of work (e.g., training, validation).
- **Artifact:** Output of a component (e.g., model, statistics).
- **Metadata store:** Tracks artifacts and lineage.

---

## 2. ExampleGen

**ExampleGen** ingests data and produces **TFRecord** examples for downstream components.

```python
from tfx.components import CsvExampleGen

example_gen = CsvExampleGen(input_base='data/')
```

**Input:** Directory with CSV files, or other supported formats (Parquet, etc.).

**Output:** TFRecord dataset in standard ExampleGen format.

**Splits:** ExampleGen can produce train/eval splits (e.g., 80/20).

---

## 3. StatisticsGen and SchemaGen

### StatisticsGen

**StatisticsGen** computes data statistics (min, max, mean, histograms) over the dataset.

```python
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
```

**Output:** Statistics artifact used by SchemaGen and ExampleValidator.

### SchemaGen

**SchemaGen** infers a schema (feature types, domains) from statistics.

```python
from tfx.components import SchemaGen

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
```

**Output:** Schema artifact defining expected feature types and constraints.

---

## 4. ExampleValidator

**ExampleValidator** checks data against the schema and previous statistics. Detects drift and anomalies.

```python
from tfx.components import ExampleValidator

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
```

**Output:** Validation results. Pipeline can be configured to fail on anomalies.

---

## 5. Transform

**Transform** applies preprocessing (scaling, encoding) and produces a **saved transform graph** for training and serving.

```python
from tfx.components import Transform

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file='preprocessing.py'
)
```

**preprocessing.py** defines a `preprocessing_fn` that takes inputs and returns a dict of transformed features.

```python
def preprocessing_fn(inputs):
    return {
        'feature': tft.scale_to_z_score(inputs['feature'])
    }
```

**Output:** Transform graph and transformed examples.

---

## 6. Trainer

**Trainer** runs model training using the transformed data and schema.

```python
from tfx.components import Trainer

trainer = Trainer(
    module_file='trainer.py',
    examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=tfx.proto.TrainArgs(num_steps=1000),
    eval_args=tfx.proto.EvalArgs(num_steps=100)
)
```

**trainer.py** defines `run_fn` that builds and trains the model.

**Output:** SavedModel artifact.

---

## 7. Evaluator and Pusher

### Evaluator

**Evaluator** validates the model against a blessing threshold (e.g., accuracy, AUC). Can compare to a baseline.

```python
from tfx.components import Evaluator

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model']
)
```

### Pusher

**Pusher** deploys the model to a serving target (e.g., TensorFlow Serving) if evaluation passes.

```python
from tfx.components import Pusher

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

---

## 8. Pipeline Orchestration

TFX pipelines run on **orchestrators** such as Apache Beam, Kubeflow Pipelines, or Airflow.

```python
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

pipeline = pipeline.Pipeline(
    pipeline_name='my_pipeline',
    pipeline_root='pipelines/',
    components=[example_gen, statistics_gen, schema_gen, example_validator,
                transform, trainer, evaluator, pusher],
    enable_cache=True
)
BeamDagRunner().run(pipeline)
```

**Key concept:** Components are DAG nodes. Artifacts flow along edges. Caching skips components whose inputs are unchanged.
