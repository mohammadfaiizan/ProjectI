# Model Interpretability in Production

## Table of Contents

1. [Introduction to Interpretability](#introduction-to-interpretability)
2. [LIME (Local Interpretable Model-Agnostic Explanations)](#lime)
3. [SHAP (SHapley Additive exPlanations)](#shap)
4. [Feature Importance](#feature-importance)
5. [Partial Dependence Plots](#partial-dependence-plots)
6. [Model Cards](#model-cards)
7. [Production Explainability Requirements](#production-explainability-requirements)
8. [Real-Time Explanations](#real-time-explanations)
9. [Explainability APIs](#explainability-apis)
10. [Key Takeaways](#key-takeaways)

## Introduction to Interpretability

Model interpretability helps understand how ML models make predictions, which is crucial for:

- **Trust**: Users and stakeholders trust model predictions
- **Debugging**: Identify and fix model issues
- **Compliance**: Meet regulatory requirements (GDPR, etc.)
- **Fairness**: Detect and mitigate bias
- **Business Insights**: Understand factors driving predictions

### Types of Interpretability

**Global Interpretability**: Understanding overall model behavior
**Local Interpretability**: Understanding individual predictions
**Post-hoc Interpretability**: Explaining trained models
**Intrinsic Interpretability**: Using inherently interpretable models

### Interpretability vs Accuracy Trade-off

Some interpretable models (linear models, decision trees) may sacrifice accuracy for interpretability. Post-hoc methods allow using complex models while maintaining interpretability.

## LIME (Local Interpretable Model-Agnostic Explanations)

LIME explains individual predictions by approximating the model locally with an interpretable model.

### Basic LIME Usage

```python
import lime
import lime.lime_tabular
import numpy as np

class LIMExplainer:
    def __init__(self, model, training_data, feature_names):
        self.model = model
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode='classification'
        )
    
    def explain_instance(self, instance, num_features=10):
        """Explain a single prediction"""
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        
        return {
            'prediction': self.model.predict([instance])[0],
            'explanation': explanation.as_list(),
            'score': explanation.score
        }
    
    def explain_batch(self, instances, num_features=10):
        """Explain multiple predictions"""
        explanations = []
        for instance in instances:
            explanation = self.explain_instance(instance, num_features)
            explanations.append(explanation)
        return explanations
```

### LIME for Text

```python
import lime.lime_text

class LIMETextExplainer:
    def __init__(self, model, class_names):
        self.model = model
        self.explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    
    def explain_instance(self, text, num_features=10):
        """Explain text prediction"""
        def predict_proba(texts):
            return self.model.predict_proba(texts)
        
        explanation = self.explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features
        )
        
        return explanation.as_list()
```

### LIME for Images

```python
import lime.lime_image
from skimage.segmentation import mark_boundaries

class LIMEImageExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime.lime_image.LimeImageExplainer()
    
    def explain_instance(self, image, top_labels=5, num_features=10):
        """Explain image prediction"""
        explanation = self.explainer.explain_instance(
            image,
            self.model.predict_proba,
            top_labels=top_labels,
            num_features=num_features
        )
        
        # Get explanation for top prediction
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=True
        )
        
        return {
            'explanation_image': mark_boundaries(temp, mask),
            'top_labels': explanation.top_labels,
            'explanation': explanation.as_list(explanation.top_labels[0])
        }
```

## SHAP (SHapley Additive exPlanations)

SHAP provides unified framework for explaining model predictions using Shapley values from game theory.

### TreeExplainer

```python
import shap

class SHAPTreeExplainer:
    def __init__(self, model, training_data):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(training_data)
    
    def explain_instance(self, instance):
        """Explain single prediction"""
        shap_values = self.explainer.shap_values([instance])
        
        return {
            'shap_values': shap_values[0],
            'base_value': self.explainer.expected_value,
            'prediction': self.model.predict([instance])[0]
        }
    
    def explain_batch(self, instances):
        """Explain batch of predictions"""
        shap_values = self.explainer.shap_values(instances)
        return shap_values
```

### KernelExplainer

```python
class SHAPKernelExplainer:
    def __init__(self, model, training_data, sample_size=100):
        self.model = model
        # Sample data for faster computation
        sample_data = training_data.sample(min(sample_size, len(training_data)))
        self.explainer = shap.KernelExplainer(
            model.predict_proba,
            sample_data
        )
    
    def explain_instance(self, instance, num_samples=100):
        """Explain single prediction"""
        shap_values = self.explainer.shap_values(
            instance,
            nsamples=num_samples
        )
        
        return {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value
        }
```

### DeepExplainer

```python
import tensorflow as tf

class SHAPDeepExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain_instance(self, instance):
        """Explain single prediction"""
        shap_values = self.explainer.shap_values([instance])
        
        return {
            'shap_values': shap_values[0],
            'base_value': self.explainer.expected_value
        }
```

### SHAP Summary Plots

```python
class SHAPVisualizer:
    def __init__(self, explainer, data):
        self.explainer = explainer
        self.data = data
        self.shap_values = explainer.shap_values(data)
    
    def summary_plot(self):
        """Create summary plot"""
        shap.summary_plot(self.shap_values, self.data)
    
    def waterfall_plot(self, instance_index):
        """Create waterfall plot for instance"""
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_index],
                base_values=self.explainer.expected_value,
                data=self.data.iloc[instance_index]
            )
        )
    
    def dependence_plot(self, feature_index):
        """Create dependence plot"""
        shap.dependence_plot(
            feature_index,
            self.shap_values,
            self.data
        )
```

## Feature Importance

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

class PermutationImportance:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def calculate_importance(self, n_repeats=10):
        """Calculate permutation importance"""
        result = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=42
        )
        
        importance_df = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
```

### Tree-Based Importance

```python
class TreeFeatureImportance:
    def __init__(self, model):
        self.model = model
    
    def get_importance(self):
        """Get feature importance from tree-based model"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        return importances
    
    def plot_importance(self, feature_names, top_n=20):
        """Plot feature importance"""
        importances = self.get_importance()
        
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance")
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        return plt.gcf()
```

## Partial Dependence Plots

Partial Dependence Plots show the marginal effect of features on predictions.

### Implementation

```python
from sklearn.inspection import PartialDependenceDisplay

class PartialDependenceAnalyzer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
    
    def plot_partial_dependence(self, features, grid_resolution=50):
        """Create partial dependence plots"""
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_train,
            features,
            grid_resolution=grid_resolution
        )
        return display
    
    def partial_dependence_values(self, feature_index, grid_resolution=50):
        """Get partial dependence values"""
        from sklearn.inspection import partial_dependence
        
        pd_result = partial_dependence(
            self.model,
            self.X_train,
            features=[feature_index],
            grid_resolution=grid_resolution
        )
        
        return {
            'feature_values': pd_result['grid_values'][0],
            'partial_dependence': pd_result['average'][0]
        }
```

## Model Cards

Model cards document model performance, limitations, and intended use.

### Model Card Structure

```python
class ModelCard:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.card = {
            'model_details': {},
            'intended_use': {},
            'factors': {},
            'metrics': {},
            'evaluation_data': {},
            'training_data': {},
            'quantitative_analyses': {},
            'ethical_considerations': {},
            'caveats_and_recommendations': {}
        }
    
    def add_model_details(self, description, version, owners):
        """Add model details"""
        self.card['model_details'] = {
            'name': self.model_name,
            'version': self.model_version,
            'description': description,
            'owners': owners
        }
    
    def add_intended_use(self, primary_uses, out_of_scope_uses):
        """Add intended use cases"""
        self.card['intended_use'] = {
            'primary_uses': primary_uses,
            'out_of_scope_uses': out_of_scope_uses
        }
    
    def add_metrics(self, performance_metrics, fairness_metrics):
        """Add performance metrics"""
        self.card['metrics'] = {
            'performance': performance_metrics,
            'fairness': fairness_metrics
        }
    
    def add_evaluation_data(self, dataset_description, performance_by_group):
        """Add evaluation data information"""
        self.card['evaluation_data'] = {
            'dataset': dataset_description,
            'performance_by_group': performance_by_group
        }
    
    def to_json(self):
        """Export model card as JSON"""
        return json.dumps(self.card, indent=2)
```

## Production Explainability Requirements

### Regulatory Compliance

```python
class ExplainabilityCompliance:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
    
    def generate_explanation(self, instance, user_id=None):
        """Generate explanation for compliance"""
        explanation = self.explainer.explain_instance(instance)
        
        # Add required metadata
        explanation_metadata = {
            'explanation': explanation,
            'timestamp': time.time(),
            'user_id': user_id,
            'model_version': self.model.version,
            'compliance': {
                'gdpr_compliant': True,
                'explanation_type': 'local',
                'features_explained': len(explanation['explanation'])
            }
        }
        
        return explanation_metadata
    
    def audit_trail(self, explanations):
        """Maintain audit trail of explanations"""
        audit_log = {
            'explanations': explanations,
            'retention_period_days': 365,
            'encrypted': True
        }
        return audit_log
```

### Performance Requirements

```python
class ExplainabilityPerformance:
    def __init__(self, explainer, max_latency_ms=100):
        self.explainer = explainer
        self.max_latency_ms = max_latency_ms
    
    def explain_with_timeout(self, instance):
        """Generate explanation with timeout"""
        start_time = time.time()
        
        try:
            explanation = self.explainer.explain_instance(instance)
            latency_ms = (time.time() - start_time) * 1000
            
            if latency_ms > self.max_latency_ms:
                # Use cached or simplified explanation
                explanation = self.get_cached_explanation(instance)
            
            return {
                'explanation': explanation,
                'latency_ms': latency_ms,
                'within_sla': latency_ms <= self.max_latency_ms
            }
        except TimeoutError:
            return {
                'explanation': None,
                'error': 'Timeout',
                'latency_ms': self.max_latency_ms
            }
```

## Real-Time Explanations

### Caching Explanations

```python
from functools import lru_cache
import hashlib

class CachedExplainer:
    def __init__(self, explainer, cache_size=10000):
        self.explainer = explainer
        self.cache = {}
        self.cache_size = cache_size
    
    def explain_instance(self, instance):
        """Explain with caching"""
        # Create hash of instance
        instance_hash = hashlib.md5(
            str(instance).encode()
        ).hexdigest()
        
        # Check cache
        if instance_hash in self.cache:
            return self.cache[instance_hash]
        
        # Generate explanation
        explanation = self.explainer.explain_instance(instance)
        
        # Cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[instance_hash] = explanation
        else:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.cache[instance_hash] = explanation
        
        return explanation
```

### Batch Explanations

```python
class BatchExplainer:
    def __init__(self, explainer, batch_size=100):
        self.explainer = explainer
        self.batch_size = batch_size
    
    def explain_batch(self, instances):
        """Explain batch of instances"""
        explanations = []
        
        for i in range(0, len(instances), self.batch_size):
            batch = instances[i:i+self.batch_size]
            batch_explanations = [
                self.explainer.explain_instance(instance)
                for instance in batch
            ]
            explanations.extend(batch_explanations)
        
        return explanations
```

## Explainability APIs

### REST API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
explainer = LIMExplainer(model, training_data, feature_names)

@app.route('/explain', methods=['POST'])
def explain():
    """Explain prediction endpoint"""
    data = request.get_json()
    instance = np.array(data['features'])
    
    explanation = explainer.explain_instance(instance)
    
    return jsonify({
        'prediction': explanation['prediction'],
        'explanation': explanation['explanation'],
        'confidence': explanation.get('confidence', None)
    })

@app.route('/explain/batch', methods=['POST'])
def explain_batch():
    """Batch explanation endpoint"""
    data = request.get_json()
    instances = [np.array(inst) for inst in data['instances']]
    
    explanations = explainer.explain_batch(instances)
    
    return jsonify({
        'explanations': explanations
    })
```

### GraphQL API

```python
import graphene

class Explanation(graphene.ObjectType):
    prediction = graphene.Float()
    features = graphene.List(graphene.String)
    contributions = graphene.List(graphene.Float())

class Query(graphene.ObjectType):
    explain = graphene.Field(
        Explanation,
        features=graphene.List(graphene.Float)
    )
    
    def resolve_explain(self, info, features):
        instance = np.array(features)
        explanation = explainer.explain_instance(instance)
        
        return Explanation(
            prediction=explanation['prediction'],
            features=[f[0] for f in explanation['explanation']],
            contributions=[f[1] for f in explanation['explanation']]
        )

schema = graphene.Schema(query=Query)
```

## Key Takeaways

- Model interpretability builds trust and enables debugging and compliance
- LIME provides local, model-agnostic explanations for individual predictions
- SHAP offers unified framework with theoretical guarantees for explanations
- Feature importance identifies which features drive model predictions
- Partial dependence plots show marginal effects of features
- Model cards document model behavior, limitations, and intended use
- Production explainability must meet performance and compliance requirements
- Real-time explanations require caching and optimization for low latency
- Explainability APIs enable integration into applications and workflows
- Balancing interpretability with model complexity and performance is crucial
