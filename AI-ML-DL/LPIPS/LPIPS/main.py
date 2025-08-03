"""
LPIPS Main Application
======================

Complete LPIPS implementation workflow demonstrating training, evaluation, and analysis
of Learned Perceptual Image Patch Similarity using JND datasets.

This main script provides:
- Complete LPIPS training pipeline
- JND dataset integration
- Model evaluation and comparison
- Traditional metrics comparison
- Comprehensive analysis and visualization
- Production-ready model deployment

Usage:
    python main.py --mode train --data_dir ./LPIPS_Data --backbone vgg
    python main.py --mode evaluate --model_path ./checkpoints/best_model.pth
    python main.py --mode compare --data_dir ./LPIPS_Data

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import os
import sys
import time
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import LPIPS modules
from lpips_model import LPIPS, create_lpips_model, compare_lpips_variants
from data_loader import LPIPSDataModule, JNDDataset, create_synthetic_jnd_dataset, visualize_jnd_dataset
from trainer import LPIPSTrainer, create_lpips_trainer
from evaluation_metrics import LPIPSEvaluator, LPIPSVisualizer, TraditionalMetrics, create_evaluation_report


class LPIPSApplication:
    """
    Main LPIPS application class handling all workflows
    """
    
    def __init__(self, args):
        """Initialize LPIPS application with command line arguments"""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        self.logs_dir = self.output_dir / 'logs'
        
        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"LPIPS Application initialized:")
        print(f"  Mode: {args.mode}")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {self.output_dir}")
    
    def run(self):
        """Run the application based on the specified mode"""
        
        if self.args.mode == 'train':
            self.train_lpips()
        elif self.args.mode == 'evaluate':
            self.evaluate_lpips()
        elif self.args.mode == 'compare':
            self.compare_architectures()
        elif self.args.mode == 'demo':
            self.run_demo()
        elif self.args.mode == 'analyze':
            self.analyze_dataset()
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
    
    def train_lpips(self):
        """Train LPIPS model on JND dataset"""
        
        print("="*60)
        print("TRAINING LPIPS MODEL")
        print("="*60)
        
        # Check if data directory exists
        if not os.path.exists(self.args.data_dir):
            print(f"Data directory {self.args.data_dir} not found!")
            print("Creating synthetic dataset for demonstration...")
            create_synthetic_jnd_dataset(self.args.data_dir, num_samples=1000)
        
        # Create trainer
        experiment_name = f"lpips_{self.args.backbone}_{int(time.time())}"
        trainer = create_lpips_trainer(
            backbone=self.args.backbone,
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            experiment_name=experiment_name
        )
        
        # Override checkpoint and log directories
        trainer.checkpoint_dir = self.checkpoint_dir / experiment_name
        trainer.log_dir = self.logs_dir / experiment_name
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if resuming
        start_epoch = 0
        if self.args.resume and self.args.checkpoint_path:
            if os.path.exists(self.args.checkpoint_path):
                start_epoch = trainer.load_checkpoint(self.args.checkpoint_path)
                print(f"Resumed training from epoch {start_epoch}")
        
        # Train model
        print(f"Starting training for {self.args.epochs} epochs...")
        train_history = trainer.train(
            num_epochs=self.args.epochs,
            validate_every=1,
            save_every=10,
            early_stopping_patience=self.args.patience
        )
        
        # Plot training history
        plot_path = self.results_dir / f"{experiment_name}_training_history.png"
        trainer.plot_training_history(save_path=str(plot_path))
        
        # Evaluate final model
        print("\nEvaluating trained model...")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        eval_path = self.results_dir / f"{experiment_name}_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Create comprehensive report
        report_path = self.results_dir / f"{experiment_name}_training_report.json"
        self._create_training_report(trainer, eval_results, train_history, report_path)
        
        print(f"\nTraining completed!")
        print(f"  Best model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")
        print(f"  Training history plot: {plot_path}")
        print(f"  Evaluation results: {eval_path}")
        print(f"  Training report: {report_path}")
    
    def evaluate_lpips(self):
        """Evaluate pre-trained LPIPS model"""
        
        print("="*60)
        print("EVALUATING LPIPS MODEL")
        print("="*60)
        
        # Load model
        if not self.args.model_path or not os.path.exists(self.args.model_path):
            print("Model path not specified or doesn't exist. Using pretrained model...")
            model = create_lpips_model(self.args.backbone, pretrained=True)
        else:
            # Load from checkpoint
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            model_config = checkpoint.get('model_config', {})
            backbone = model_config.get('backbone', self.args.backbone)
            
            model = create_lpips_model(backbone, pretrained=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {self.args.model_path}")
        
        # Create data module
        data_module = LPIPSDataModule(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size
        )
        data_module.setup()
        
        # Create evaluator
        evaluator = LPIPSEvaluator(model, self.device)
        
        # Run evaluation
        print("Running comprehensive evaluation...")
        eval_results = evaluator.evaluate_2afc_dataset(data_module.test_dataloader())
        
        # Create visualizations
        visualizer = LPIPSVisualizer()
        
        # Correlation comparison plot
        if 'correlations' in eval_results:
            corr_plot_path = self.results_dir / 'correlation_comparison.png'
            visualizer.plot_correlation_comparison(
                eval_results['correlations'], 
                save_path=str(corr_plot_path)
            )
        
        # Save evaluation results
        eval_path = self.results_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_eval_results = self._convert_to_json_serializable(eval_results)
            json.dump(json_eval_results, f, indent=2)
        
        # Create evaluation report
        report_path = self.results_dir / 'evaluation_report.json'
        create_evaluation_report(eval_results, str(report_path))
        
        # Print summary
        print(f"\nEvaluation completed!")
        print(f"  2AFC Accuracy: {eval_results['accuracy']:.4f}")
        
        if 'correlations' in eval_results and 'lpips' in eval_results['correlations']:
            lpips_corr = eval_results['correlations']['lpips']['pearson_r']
            print(f"  LPIPS-Human Correlation: {abs(lpips_corr):.4f}")
        
        print(f"  Results saved to: {eval_path}")
        print(f"  Report saved to: {report_path}")
    
    def compare_architectures(self):
        """Compare different LPIPS architectures"""
        
        print("="*60)
        print("COMPARING LPIPS ARCHITECTURES")
        print("="*60)
        
        # Create models for comparison
        architectures = {}
        
        for backbone in ['alexnet', 'vgg', 'squeezenet']:
            try:
                model = create_lpips_model(backbone, pretrained=True)
                architectures[f'LPIPS-{backbone}'] = model
                print(f"Created {backbone} model")
            except Exception as e:
                print(f"Failed to create {backbone} model: {e}")
        
        if not architectures:
            print("No models created for comparison!")
            return
        
        # Create data module
        data_module = LPIPSDataModule(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size
        )
        data_module.setup()
        
        # Create evaluator (use first model as base)
        base_model = list(architectures.values())[0]
        evaluator = LPIPSEvaluator(base_model, self.device)
        
        # Compare architectures
        print("Comparing architectures...")
        comparison_results = evaluator.compare_architectures(
            architectures, 
            data_module.test_dataloader()
        )
        
        # Create visualizations
        visualizer = LPIPSVisualizer()
        
        # Architecture comparison plot
        comp_plot_path = self.results_dir / 'architecture_comparison.png'
        visualizer.plot_architecture_comparison(
            comparison_results,
            save_path=str(comp_plot_path)
        )
        
        # Save comparison results
        comp_path = self.results_dir / 'architecture_comparison.json'
        with open(comp_path, 'w') as f:
            json_results = self._convert_to_json_serializable(comparison_results)
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print(f"\nArchitecture comparison completed!")
        
        summary = comparison_results['summary']
        print(f"  Best accuracy: {summary['accuracy_ranking'][0][0]} ({summary['accuracy_ranking'][0][1]:.4f})")
        print(f"  Best correlation: {summary['correlation_ranking'][0][0]} ({summary['correlation_ranking'][0][1]:.4f})")
        print(f"  Best overall: {summary['best_overall']}")
        
        print(f"  Results saved to: {comp_path}")
        print(f"  Visualization saved to: {comp_plot_path}")
    
    def run_demo(self):
        """Run complete LPIPS demo with synthetic data"""
        
        print("="*60)
        print("LPIPS COMPLETE DEMO")
        print("="*60)
        
        # Create synthetic dataset
        demo_data_dir = self.output_dir / 'demo_data'
        print("Creating synthetic JND dataset...")
        create_synthetic_jnd_dataset(str(demo_data_dir), num_samples=500)
        
        # Visualize dataset
        print("Visualizing dataset samples...")
        try:
            demo_dataset = JNDDataset(str(demo_data_dir / 'train'))
            visualize_jnd_dataset(demo_dataset, num_samples=3)
        except Exception as e:
            print(f"Could not visualize dataset: {e}")
        
        # Train small model
        print("Training LPIPS model (demo - few epochs)...")
        trainer = create_lpips_trainer(
            backbone='squeezenet',  # Fastest for demo
            data_dir=str(demo_data_dir),
            batch_size=16,
            learning_rate=1e-3,
            experiment_name='demo'
        )
        
        # Override directories
        trainer.checkpoint_dir = self.checkpoint_dir / 'demo'
        trainer.log_dir = self.logs_dir / 'demo'
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Train for few epochs
        train_history = trainer.train(num_epochs=5, validate_every=1)
        
        # Plot training history
        plot_path = self.results_dir / 'demo_training_history.png'
        trainer.plot_training_history(save_path=str(plot_path))
        
        # Evaluate model
        print("Evaluating trained model...")
        eval_results = trainer.evaluate()
        
        # Compare with traditional metrics
        print("Comparing with traditional metrics...")
        
        # Create simple comparison
        data_module = LPIPSDataModule(str(demo_data_dir), batch_size=16)
        data_module.setup()
        
        evaluator = LPIPSEvaluator(trainer.model, self.device)
        comparison_results = evaluator.evaluate_2afc_dataset(data_module.test_dataloader())
        
        # Visualize results
        visualizer = LPIPSVisualizer()
        
        if 'correlations' in comparison_results:
            corr_plot_path = self.results_dir / 'demo_correlations.png'
            visualizer.plot_correlation_comparison(
                comparison_results['correlations'],
                save_path=str(corr_plot_path)
            )
        
        # Save demo results
        demo_results = {
            'training_history': train_history,
            'evaluation_results': eval_results,
            'comparison_results': self._convert_to_json_serializable(comparison_results)
        }
        
        demo_path = self.results_dir / 'demo_results.json'
        with open(demo_path, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"\nDemo completed!")
        print(f"  Final accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Results saved to: {demo_path}")
        print(f"  Training plot: {plot_path}")
    
    def analyze_dataset(self):
        """Analyze JND dataset characteristics"""
        
        print("="*60)
        print("ANALYZING JND DATASET")
        print("="*60)
        
        # Create data module
        data_module = LPIPSDataModule(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size
        )
        data_module.setup()
        
        # Get dataset info
        info = data_module.get_dataset_info()
        
        print("Dataset Information:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Train samples: {info['train_samples']}")
        print(f"  Validation samples: {info['val_samples']}")
        print(f"  Test samples: {info['test_samples']}")
        
        # Analyze category distribution if available
        try:
            train_dataset = JNDDataset(
                data_dir=os.path.join(self.args.data_dir, 'train') 
                if os.path.exists(os.path.join(self.args.data_dir, 'train')) 
                else self.args.data_dir
            )
            
            categories = train_dataset.get_category_distribution()
            
            print("\nCategory Distribution:")
            for category, count in categories.items():
                print(f"  {category}: {count} samples")
            
            # Visualize some samples
            print("\nVisualizing dataset samples...")
            visualize_jnd_dataset(train_dataset, num_samples=5)
            
        except Exception as e:
            print(f"Could not analyze dataset details: {e}")
        
        # Save analysis
        analysis_results = {
            'dataset_info': info,
            'analysis_timestamp': time.time()
        }
        
        analysis_path = self.results_dir / 'dataset_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\nDataset analysis saved to: {analysis_path}")
    
    def _create_training_report(self, 
                              trainer: LPIPSTrainer, 
                              eval_results: Dict[str, Any],
                              train_history: Dict[str, Any],
                              report_path: Path):
        """Create comprehensive training report"""
        
        report = {
            'experiment_info': {
                'backbone': trainer.model.backbone_name,
                'device': str(trainer.device),
                'experiment_name': trainer.experiment_name,
                'timestamp': time.time()
            },
            'model_info': trainer.model.get_model_info(),
            'training_config': {
                'epochs_trained': len(train_history['loss']),
                'best_val_accuracy': trainer.best_val_accuracy,
                'best_val_correlation': trainer.best_val_correlation,
                'final_learning_rate': train_history['learning_rates'][-1] if train_history['learning_rates'] else 0
            },
            'training_history': train_history,
            'evaluation_results': eval_results,
            'key_achievements': [
                f"Achieved {trainer.best_val_accuracy:.4f} validation accuracy",
                f"Achieved {trainer.best_val_correlation:.4f} human correlation",
                f"Trained for {len(train_history['loss'])} epochs"
            ]
        }
        
        # Add comparison with traditional metrics if available
        if 'correlations' in eval_results:
            correlations = eval_results['correlations']
            
            if 'lpips' in correlations:
                lpips_corr = abs(correlations['lpips'].get('pearson_r', 0))
                report['key_achievements'].append(f"LPIPS correlation: {lpips_corr:.4f}")
            
            # Find best traditional metric
            traditional_correlations = {k: v for k, v in correlations.items() if k != 'lpips'}
            if traditional_correlations:
                best_traditional = max(traditional_correlations.items(), 
                                     key=lambda x: abs(x[1].get('pearson_r', 0)))
                best_corr = abs(best_traditional[1].get('pearson_r', 0))
                
                report['key_achievements'].append(
                    f"Best traditional metric: {best_traditional[0]} ({best_corr:.4f})"
                )
                
                if 'lpips' in correlations:
                    improvement = lpips_corr - best_corr
                    report['key_achievements'].append(
                        f"LPIPS improvement: {improvement:.4f}"
                    )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='LPIPS: Learned Perceptual Image Patch Similarity')
    
    # Main arguments
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'compare', 'demo', 'analyze'],
                       help='Operation mode')
    
    parser.add_argument('--data_dir', type=str, default='./LPIPS_Data',
                       help='Path to JND dataset directory')
    
    parser.add_argument('--output_dir', type=str, default='./lpips_output',
                       help='Output directory for results and checkpoints')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vgg',
                       choices=['alexnet', 'vgg', 'squeezenet'],
                       help='Backbone network for LPIPS')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model (for evaluation)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training/evaluation')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for resuming training')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("LPIPS: LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Create and run application
    app = LPIPSApplication(args)
    app.run()


if __name__ == "__main__":
    main()