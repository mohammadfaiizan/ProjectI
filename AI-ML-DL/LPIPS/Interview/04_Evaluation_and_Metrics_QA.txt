LPIPS Interview Questions & Answers - Evaluation and Metrics
============================================================

This file contains questions about LPIPS evaluation methodologies, metrics,
benchmarking, and comparison with traditional image quality metrics.

============================================================

Q1: How do you evaluate LPIPS model performance comprehensively?

A1: Comprehensive LPIPS evaluation requires multiple metrics and methodologies:

**Primary Evaluation Metrics**:

1. **2AFC Accuracy**:
```python
def compute_2afc_accuracy(lpips_model, test_loader):
    """Compute Two-Alternative Forced Choice accuracy"""
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for ref_imgs, img1s, img2s, judgments in test_loader:
            # Compute LPIPS distances
            dist1 = lpips_model(ref_imgs, img1s)
            dist2 = lpips_model(ref_imgs, img2s)
            
            # Make predictions (0 if img1 closer, 1 if img2 closer)
            predictions = (dist1 > dist2).long()
            
            # Count correct predictions
            correct = (predictions.squeeze() == judgments).sum().item()
            correct_predictions += correct
            total_samples += len(judgments)
    
    accuracy = correct_predictions / total_samples
    return accuracy
```

2. **Correlation with Human Judgment**:
```python
def compute_human_correlation(lpips_distances, human_judgments):
    """Compute correlation metrics with human perception"""
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    # Ensure same length
    min_len = min(len(lpips_distances), len(human_judgments))
    lpips_subset = lpips_distances[:min_len]
    human_subset = human_judgments[:min_len]
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(lpips_subset, human_subset)
    spearman_r, spearman_p = spearmanr(lpips_subset, human_subset)
    kendall_tau, kendall_p = kendalltau(lpips_subset, human_subset)
    
    return {
        'pearson': {'correlation': pearson_r, 'p_value': pearson_p},
        'spearman': {'correlation': spearman_r, 'p_value': spearman_p},
        'kendall': {'correlation': kendall_tau, 'p_value': kendall_p}
    }
```

3. **Ranking Quality Assessment**:
```python
def evaluate_ranking_quality(lpips_model, image_sets):
    """Evaluate quality of perceptual ranking"""
    ranking_metrics = []
    
    for reference, candidates, human_rankings in image_sets:
        # Compute LPIPS distances
        lpips_distances = []
        for candidate in candidates:
            distance = lpips_model(reference.unsqueeze(0), candidate.unsqueeze(0))
            lpips_distances.append(distance.item())
        
        # LPIPS ranking (ascending distance = better similarity)
        lpips_ranking = np.argsort(lpips_distances)
        
        # Compute ranking correlation
        from scipy.stats import spearmanr
        rank_correlation, _ = spearmanr(lpips_ranking, human_rankings)
        ranking_metrics.append(rank_correlation)
    
    return {
        'mean_ranking_correlation': np.mean(ranking_metrics),
        'std_ranking_correlation': np.std(ranking_metrics),
        'individual_correlations': ranking_metrics
    }
```

**Comprehensive Evaluation Framework**:
```python
class LPIPSEvaluationSuite:
    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        self.evaluation_results = {}
    
    def run_full_evaluation(self, test_datasets):
        """Run complete evaluation suite"""
        results = {}
        
        # 1. 2AFC Evaluation
        results['2afc_accuracy'] = self.evaluate_2afc(test_datasets['2afc'])
        
        # 2. Correlation Analysis
        results['correlations'] = self.evaluate_correlations(test_datasets['correlation'])
        
        # 3. Ranking Quality
        results['ranking'] = self.evaluate_ranking(test_datasets['ranking'])
        
        # 4. Cross-dataset Generalization
        results['generalization'] = self.evaluate_generalization(test_datasets['cross_domain'])
        
        # 5. Computational Efficiency
        results['efficiency'] = self.evaluate_efficiency()
        
        # 6. Robustness Analysis
        results['robustness'] = self.evaluate_robustness(test_datasets['robustness'])
        
        return results
```

============================================================

Q2: How do you compare LPIPS performance with traditional image quality metrics?

A2: Systematic comparison requires standardized evaluation protocols:

**Traditional Metrics Implementation**:
```python
class TraditionalMetricsEvaluator:
    def __init__(self):
        self.metrics = {
            'psnr': self.compute_psnr,
            'ssim': self.compute_ssim,
            'ms_ssim': self.compute_ms_ssim,
            'lpips_alex': None,  # Will be set externally
            'lpips_vgg': None,
            'lpips_squeeze': None
        }
    
    def compute_psnr(self, img1, img2, max_val=1.0):
        """Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, img1, img2, window_size=11, data_range=1.0):
        """Structural Similarity Index"""
        # Convert to grayscale
        if img1.shape[0] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img1_gray = (img1 * weights).sum(dim=0, keepdim=True)
            img2_gray = (img2 * weights).sum(dim=0, keepdim=True)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # SSIM calculation
        mu1 = img1_gray.mean()
        mu2 = img2_gray.mean()
        
        sigma1_sq = ((img1_gray - mu1) ** 2).mean()
        sigma2_sq = ((img2_gray - mu2) ** 2).mean()
        sigma12 = ((img1_gray - mu1) * (img2_gray - mu2)).mean()
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.item()
    
    def compute_ms_ssim(self, img1, img2, weights=None):
        """Multi-Scale Structural Similarity Index"""
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        
        levels = len(weights)
        ms_ssim_val = 1.0
        
        for i in range(levels):
            ssim = self.compute_ssim(img1, img2)
            
            if i == levels - 1:
                ms_ssim_val *= ssim ** weights[i]
            else:
                # Downsample for next level
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
                ms_ssim_val *= ssim ** weights[i]
        
        return ms_ssim_val

def compare_all_metrics(image_pairs, human_judgments, lpips_models):
    """Comprehensive comparison of all metrics"""
    evaluator = TraditionalMetricsEvaluator()
    
    # Set LPIPS models
    evaluator.metrics['lpips_alex'] = lpips_models['alexnet']
    evaluator.metrics['lpips_vgg'] = lpips_models['vgg']
    evaluator.metrics['lpips_squeeze'] = lpips_models['squeezenet']
    
    all_results = {}
    
    for metric_name in evaluator.metrics.keys():
        print(f"Evaluating {metric_name}...")
        
        if 'lpips' in metric_name:
            # LPIPS evaluation
            model = evaluator.metrics[metric_name]
            distances = []
            
            for img1, img2 in image_pairs:
                with torch.no_grad():
                    distance = model(img1.unsqueeze(0), img2.unsqueeze(0))
                    distances.append(distance.item())
            
            # For LPIPS, higher distance = less similar
            metric_values = distances
            
        else:
            # Traditional metrics
            metric_func = evaluator.metrics[metric_name]
            metric_values = []
            
            for img1, img2 in image_pairs:
                value = metric_func(img1, img2)
                metric_values.append(value)
            
            # For SSIM/PSNR, invert for correlation (higher = less similar)
            if metric_name in ['ssim', 'ms_ssim', 'psnr']:
                metric_values = [-v for v in metric_values]
        
        # Compute correlations
        correlations = compute_human_correlation(metric_values, human_judgments)
        
        all_results[metric_name] = {
            'values': metric_values,
            'correlations': correlations,
            'mean_correlation': abs(correlations['pearson']['correlation'])
        }
    
    return all_results
```

**Visualization and Analysis**:
```python
def create_comparison_visualization(comparison_results):
    """Create comprehensive comparison visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract correlation values
    metrics = list(comparison_results.keys())
    correlations = [comparison_results[m]['mean_correlation'] for m in metrics]
    
    # 1. Bar plot of correlations
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, correlations, color=['red' if 'lpips' in m else 'blue' for m in metrics])
    plt.title('Human Correlation Comparison: LPIPS vs Traditional Metrics')
    plt.ylabel('Absolute Pearson Correlation')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Scatter plot matrix
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    human_judgments = list(range(len(comparison_results['lpips_vgg']['values'])))  # Placeholder
    
    for idx, metric in enumerate(metrics[:6]):  # Show top 6 metrics
        if idx < len(axes):
            values = comparison_results[metric]['values']
            corr = comparison_results[metric]['mean_correlation']
            
            axes[idx].scatter(values, human_judgments, alpha=0.6)
            axes[idx].set_title(f'{metric.upper()}\nCorrelation: {corr:.3f}')
            axes[idx].set_xlabel('Metric Value')
            axes[idx].set_ylabel('Human Judgment')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

============================================================

Q3: How do you evaluate LPIPS robustness across different image domains and distortions?

A3: Robustness evaluation tests LPIPS performance across various conditions:

**Domain Robustness Testing**:
```python
class DomainRobustnessEvaluator:
    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        
    def evaluate_cross_domain_performance(self, domain_datasets):
        """Evaluate performance across different image domains"""
        domain_results = {}
        
        domains = ['natural', 'faces', 'medical', 'synthetic', 'artistic']
        
        for domain in domains:
            if domain in domain_datasets:
                print(f"Evaluating {domain} domain...")
                
                dataset = domain_datasets[domain]
                accuracy = self.compute_domain_accuracy(dataset)
                correlation = self.compute_domain_correlation(dataset)
                
                domain_results[domain] = {
                    'accuracy': accuracy,
                    'correlation': correlation,
                    'sample_count': len(dataset)
                }
        
        # Compute domain variance
        accuracies = [results['accuracy'] for results in domain_results.values()]
        correlations = [results['correlation'] for results in domain_results.values()]
        
        domain_results['summary'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'domain_variance': np.var(accuracies)
        }
        
        return domain_results
    
    def compute_domain_accuracy(self, dataset):
        """Compute 2AFC accuracy for specific domain"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for ref_img, img1, img2, judgment in dataset:
                dist1 = self.lpips_model(ref_img.unsqueeze(0), img1.unsqueeze(0))
                dist2 = self.lpips_model(ref_img.unsqueeze(0), img2.unsqueeze(0))
                
                prediction = 0 if dist1 < dist2 else 1
                correct += int(prediction == judgment)
                total += 1
        
        return correct / total if total > 0 else 0.0
```

**Distortion Robustness Testing**:
```python
class DistortionRobustnessEvaluator:
    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        self.distortion_functions = {
            'gaussian_noise': self.add_gaussian_noise,
            'gaussian_blur': self.add_gaussian_blur,
            'jpeg_compression': self.simulate_jpeg,
            'salt_pepper': self.add_salt_pepper_noise,
            'motion_blur': self.add_motion_blur,
            'contrast_change': self.change_contrast,
            'brightness_change': self.change_brightness
        }
    
    def evaluate_distortion_robustness(self, clean_dataset, distortion_levels):
        """Evaluate robustness to various distortions"""
        results = {}
        
        for distortion_type, distortion_func in self.distortion_functions.items():
            print(f"Testing {distortion_type} robustness...")
            
            distortion_results = []
            
            for level in distortion_levels:
                level_accuracy = self.test_distortion_level(
                    clean_dataset, distortion_func, level
                )
                distortion_results.append({
                    'level': level,
                    'accuracy': level_accuracy
                })
            
            results[distortion_type] = distortion_results
        
        return results
    
    def test_distortion_level(self, dataset, distortion_func, level):
        """Test LPIPS accuracy at specific distortion level"""
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for ref_img, img1, img2, judgment in dataset:
                # Apply distortion to all images
                ref_distorted = distortion_func(ref_img, level)
                img1_distorted = distortion_func(img1, level)
                img2_distorted = distortion_func(img2, level)
                
                # Compute LPIPS on distorted images
                dist1 = self.lpips_model(ref_distorted.unsqueeze(0), 
                                       img1_distorted.unsqueeze(0))
                dist2 = self.lpips_model(ref_distorted.unsqueeze(0), 
                                       img2_distorted.unsqueeze(0))
                
                prediction = 0 if dist1 < dist2 else 1
                correct_predictions += int(prediction == judgment)
                total_samples += 1
        
        return correct_predictions / total_samples
    
    def add_gaussian_noise(self, image, noise_level):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * noise_level
        return torch.clamp(image + noise, 0, 1)
    
    def add_gaussian_blur(self, image, blur_sigma):
        """Apply Gaussian blur to image"""
        # Simple blur approximation
        kernel_size = int(6 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, blur_sigma)
        
        # Apply convolution
        blurred = F.conv2d(image.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), 
                          padding=kernel_size//2)
        
        return blurred.squeeze(0)
    
    def create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g2d = g1d[:, None] * g1d[None, :]
        
        return g2d / g2d.sum()
```

**Statistical Significance Testing**:
```python
def statistical_significance_test(results1, results2, test_type='paired_t'):
    """Test statistical significance between two sets of results"""
    from scipy import stats
    
    if test_type == 'paired_t':
        # Paired t-test for dependent samples
        statistic, p_value = stats.ttest_rel(results1, results2)
        
    elif test_type == 'wilcoxon':
        # Non-parametric Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(results1, results2)
        
    elif test_type == 'mann_whitney':
        # Mann-Whitney U test for independent samples
        statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'test_type': test_type
    }

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval"""
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return {
        'mean': np.mean(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence
    }
```

============================================================

Q4: How do you benchmark LPIPS computational efficiency and scalability?

A4: Computational benchmarking evaluates speed, memory usage, and scalability:

**Performance Benchmarking Suite**:
```python
import time
import psutil
import torch.profiler as profiler

class LPIPSPerformanceBenchmark:
    def __init__(self, lpips_models, device='cuda'):
        self.lpips_models = lpips_models  # Dict of model_name: model
        self.device = device
        self.benchmark_results = {}
    
    def run_comprehensive_benchmark(self):
        """Run complete performance benchmark suite"""
        results = {}
        
        # 1. Inference Speed Benchmark
        results['inference_speed'] = self.benchmark_inference_speed()
        
        # 2. Memory Usage Benchmark
        results['memory_usage'] = self.benchmark_memory_usage()
        
        # 3. Batch Size Scaling
        results['batch_scaling'] = self.benchmark_batch_scaling()
        
        # 4. Input Resolution Scaling
        results['resolution_scaling'] = self.benchmark_resolution_scaling()
        
        # 5. Throughput Benchmark
        results['throughput'] = self.benchmark_throughput()
        
        return results
    
    def benchmark_inference_speed(self, num_runs=100, warmup_runs=10):
        """Benchmark inference speed for different models"""
        speed_results = {}
        
        # Prepare test inputs
        img1 = torch.rand(1, 3, 224, 224).to(self.device)
        img2 = torch.rand(1, 3, 224, 224).to(self.device)
        
        for model_name, model in self.lpips_models.items():
            model.eval()
            model = model.to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(img1, img2)
            
            # Synchronize CUDA
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(img1, img2)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            fps = 1000 / avg_time
            
            speed_results[model_name] = {
                'avg_time_ms': avg_time,
                'fps': fps,
                'device': str(self.device)
            }
        
        return speed_results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage for different models"""
        memory_results = {}
        
        for model_name, model in self.lpips_models.items():
            model.eval()
            model = model.to(self.device)
            
            # Measure model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            model_size_mb = model_size / (1024 ** 2)
            
            # Measure GPU memory usage
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Test with different batch sizes
                batch_memory = {}
                
                for batch_size in [1, 4, 8, 16, 32]:
                    try:
                        img1 = torch.rand(batch_size, 3, 224, 224).to(self.device)
                        img2 = torch.rand(batch_size, 3, 224, 224).to(self.device)
                        
                        torch.cuda.reset_peak_memory_stats()
                        
                        with torch.no_grad():
                            _ = model(img1, img2)
                        
                        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        batch_memory[batch_size] = peak_memory
                        
                        # Cleanup
                        del img1, img2
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            batch_memory[batch_size] = "OOM"
                        else:
                            raise e
            
            memory_results[model_name] = {
                'model_size_mb': model_size_mb,
                'batch_memory_usage': batch_memory if self.device == 'cuda' else None
            }
        
        return memory_results
    
    def benchmark_batch_scaling(self):
        """Test how performance scales with batch size"""
        scaling_results = {}
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for model_name, model in self.lpips_models.items():
            model.eval()
            model = model.to(self.device)
            
            batch_results = []
            
            for batch_size in batch_sizes:
                try:
                    img1 = torch.rand(batch_size, 3, 224, 224).to(self.device)
                    img2 = torch.rand(batch_size, 3, 224, 224).to(self.device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(img1, img2)
                    
                    # Benchmark
                    num_runs = 20
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(num_runs):
                            _ = model(img1, img2)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    time_per_sample = (total_time / num_runs) / batch_size * 1000  # ms per sample
                    throughput = (batch_size * num_runs) / total_time  # samples per second
                    
                    batch_results.append({
                        'batch_size': batch_size,
                        'time_per_sample_ms': time_per_sample,
                        'throughput_fps': throughput,
                        'total_time_s': total_time
                    })
                    
                    # Cleanup
                    del img1, img2
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        batch_results.append({
                            'batch_size': batch_size,
                            'status': 'OOM'
                        })
                    else:
                        raise e
            
            scaling_results[model_name] = batch_results
        
        return scaling_results
    
    def benchmark_resolution_scaling(self):
        """Test performance scaling with input resolution"""
        resolution_results = {}
        resolutions = [128, 224, 256, 384, 512]
        
        for model_name, model in self.lpips_models.items():
            model.eval()
            model = model.to(self.device)
            
            resolution_results[model_name] = []
            
            for resolution in resolutions:
                try:
                    img1 = torch.rand(1, 3, resolution, resolution).to(self.device)
                    img2 = torch.rand(1, 3, resolution, resolution).to(self.device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(img1, img2)
                    
                    # Benchmark
                    num_runs = 20
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(num_runs):
                            _ = model(img1, img2)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / num_runs * 1000  # ms
                    
                    resolution_results[model_name].append({
                        'resolution': resolution,
                        'avg_time_ms': avg_time,
                        'pixels': resolution * resolution,
                        'time_per_pixel_ns': (avg_time * 1e6) / (resolution * resolution)
                    })
                    
                    # Cleanup
                    del img1, img2
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        resolution_results[model_name].append({
                            'resolution': resolution,
                            'status': 'OOM'
                        })
                    else:
                        raise e
        
        return resolution_results

def create_performance_report(benchmark_results):
    """Create comprehensive performance report"""
    report = {
        'summary': {},
        'detailed_results': benchmark_results,
        'recommendations': []
    }
    
    # Speed summary
    speed_results = benchmark_results['inference_speed']
    fastest_model = min(speed_results.keys(), 
                       key=lambda x: speed_results[x]['avg_time_ms'])
    
    report['summary']['fastest_model'] = {
        'model': fastest_model,
        'time_ms': speed_results[fastest_model]['avg_time_ms'],
        'fps': speed_results[fastest_model]['fps']
    }
    
    # Memory summary
    memory_results = benchmark_results['memory_usage']
    most_efficient = min(memory_results.keys(),
                        key=lambda x: memory_results[x]['model_size_mb'])
    
    report['summary']['most_memory_efficient'] = {
        'model': most_efficient,
        'size_mb': memory_results[most_efficient]['model_size_mb']
    }
    
    # Generate recommendations
    if speed_results[fastest_model]['avg_time_ms'] < 10:
        report['recommendations'].append("Real-time processing feasible with fastest model")
    
    if memory_results[most_efficient]['model_size_mb'] < 5:
        report['recommendations'].append("Mobile deployment feasible with most efficient model")
    
    return report
```

============================================================

Q5: How do you evaluate LPIPS performance on out-of-distribution data?

A5: Out-of-distribution (OOD) evaluation tests model generalization:

**OOD Detection and Evaluation**:
```python
class OODEvaluator:
    def __init__(self, lpips_model, training_data_stats):
        self.lpips_model = lpips_model
        self.training_stats = training_data_stats
        
    def detect_ood_samples(self, test_dataset, threshold_percentile=95):
        """Detect out-of-distribution samples"""
        ood_scores = []
        
        with torch.no_grad():
            for ref_img, img1, img2, _ in test_dataset:
                # Compute distance to training distribution
                ood_score = self.compute_ood_score(ref_img, img1, img2)
                ood_scores.append(ood_score)
        
        # Determine threshold from training data
        threshold = np.percentile(ood_scores, threshold_percentile)
        
        ood_mask = np.array(ood_scores) > threshold
        
        return {
            'ood_scores': ood_scores,
            'ood_mask': ood_mask,
            'threshold': threshold,
            'ood_ratio': ood_mask.mean()
        }
    
    def compute_ood_score(self, ref_img, img1, img2):
        """Compute OOD score based on feature statistics"""
        # Extract features from all images
        features_ref = self.lpips_model.get_features(ref_img.unsqueeze(0))
        features_img1 = self.lpips_model.get_features(img1.unsqueeze(0))
        features_img2 = self.lpips_model.get_features(img2.unsqueeze(0))
        
        total_ood_score = 0
        
        for layer_name in features_ref.keys():
            # Compute feature statistics
            feat_ref = features_ref[layer_name].flatten()
            feat_img1 = features_img1[layer_name].flatten()
            feat_img2 = features_img2[layer_name].flatten()
            
            # Compare with training statistics
            train_mean = self.training_stats[layer_name]['mean']
            train_std = self.training_stats[layer_name]['std']
            
            # Mahalanobis distance approximation
            for feat in [feat_ref, feat_img1, feat_img2]:
                normalized_feat = (feat - train_mean) / (train_std + 1e-8)
                ood_score = torch.norm(normalized_feat).item()
                total_ood_score += ood_score
        
        return total_ood_score
    
    def evaluate_ood_performance(self, id_dataset, ood_datasets):
        """Evaluate performance on in-distribution vs out-of-distribution data"""
        results = {}
        
        # Evaluate on in-distribution data
        id_accuracy = self.compute_accuracy(id_dataset)
        id_correlation = self.compute_correlation(id_dataset)
        
        results['in_distribution'] = {
            'accuracy': id_accuracy,
            'correlation': id_correlation,
            'sample_count': len(id_dataset)
        }
        
        # Evaluate on each OOD dataset
        for ood_name, ood_dataset in ood_datasets.items():
            ood_accuracy = self.compute_accuracy(ood_dataset)
            ood_correlation = self.compute_correlation(ood_dataset)
            
            results[ood_name] = {
                'accuracy': ood_accuracy,
                'correlation': ood_correlation,
                'sample_count': len(ood_dataset),
                'accuracy_drop': id_accuracy - ood_accuracy,
                'correlation_drop': id_correlation - ood_correlation
            }
        
        # Compute overall OOD performance
        ood_accuracies = [results[name]['accuracy'] for name in ood_datasets.keys()]
        ood_correlations = [results[name]['correlation'] for name in ood_datasets.keys()]
        
        results['ood_summary'] = {
            'mean_accuracy': np.mean(ood_accuracies),
            'std_accuracy': np.std(ood_accuracies),
            'mean_correlation': np.mean(ood_correlations),
            'std_correlation': np.std(ood_correlations),
            'worst_case_accuracy': min(ood_accuracies),
            'worst_case_correlation': min(ood_correlations)
        }
        
        return results

class DomainAdaptationEvaluator:
    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        
    def evaluate_domain_adaptation(self, source_domain, target_domains, adaptation_methods):
        """Evaluate domain adaptation performance"""
        results = {}
        
        # Baseline: no adaptation
        baseline_results = {}
        for target_name, target_data in target_domains.items():
            baseline_accuracy = self.compute_accuracy(target_data)
            baseline_results[target_name] = baseline_accuracy
        
        results['baseline'] = baseline_results
        
        # Test adaptation methods
        for method_name, adaptation_func in adaptation_methods.items():
            method_results = {}
            
            for target_name, target_data in target_domains.items():
                # Apply adaptation
                adapted_model = adaptation_func(self.lpips_model, source_domain, target_data)
                
                # Evaluate adapted model
                adapted_accuracy = self.compute_accuracy_with_model(adapted_model, target_data)
                improvement = adapted_accuracy - baseline_results[target_name]
                
                method_results[target_name] = {
                    'accuracy': adapted_accuracy,
                    'improvement': improvement
                }
            
            results[method_name] = method_results
        
        return results
```

**Uncertainty Quantification**:
```python
class UncertaintyQuantification:
    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        
    def monte_carlo_dropout_uncertainty(self, img1, img2, num_samples=100):
        """Estimate uncertainty using Monte Carlo Dropout"""
        # Enable dropout during inference
        self.lpips_model.train()
        
        distances = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                distance = self.lpips_model(img1, img2)
                distances.append(distance.item())
        
        # Restore evaluation mode
        self.lpips_model.eval()
        
        distances = np.array(distances)
        
        return {
            'mean': distances.mean(),
            'std': distances.std(),
            'confidence_interval_95': np.percentile(distances, [2.5, 97.5]),
            'uncertainty': distances.std()
        }
    
    def ensemble_uncertainty(self, ensemble_models, img1, img2):
        """Estimate uncertainty using model ensemble"""
        predictions = []
        
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                distance = model(img1, img2)
                predictions.append(distance.item())
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(),
            'std': predictions.std(),
            'uncertainty': predictions.std(),
            'individual_predictions': predictions
        }
    
    def calibration_analysis(self, test_dataset, confidence_levels=[0.5, 0.7, 0.8, 0.9, 0.95]):
        """Analyze model calibration"""
        calibration_results = {}
        
        uncertainties = []
        accuracies = []
        
        for ref_img, img1, img2, judgment in test_dataset:
            # Get uncertainty estimate
            uncertainty_info = self.monte_carlo_dropout_uncertainty(
                ref_img.unsqueeze(0), img1.unsqueeze(0)
            )
            
            uncertainty_info_2 = self.monte_carlo_dropout_uncertainty(
                ref_img.unsqueeze(0), img2.unsqueeze(0)
            )
            
            # Compute prediction accuracy
            dist1_mean = uncertainty_info['mean']
            dist2_mean = uncertainty_info_2['mean']
            
            prediction = 0 if dist1_mean < dist2_mean else 1
            accuracy = int(prediction == judgment)
            
            # Combined uncertainty
            combined_uncertainty = (uncertainty_info['uncertainty'] + 
                                  uncertainty_info_2['uncertainty']) / 2
            
            uncertainties.append(combined_uncertainty)
            accuracies.append(accuracy)
        
        # Analyze calibration for different confidence levels
        for confidence in confidence_levels:
            threshold = np.percentile(uncertainties, (1 - confidence) * 100)
            confident_mask = np.array(uncertainties) <= threshold
            
            if confident_mask.sum() > 0:
                confident_accuracy = np.array(accuracies)[confident_mask].mean()
                coverage = confident_mask.mean()
                
                calibration_results[confidence] = {
                    'expected_accuracy': confidence,
                    'actual_accuracy': confident_accuracy,
                    'coverage': coverage,
                    'calibration_error': abs(confidence - confident_accuracy)
                }
        
        return calibration_results
```

This comprehensive evaluation framework provides thorough assessment of LPIPS performance across multiple dimensions.