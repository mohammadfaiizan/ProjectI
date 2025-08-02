import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime

# Kubernetes PyTorch Scaling and Orchestration
# Note: This demonstrates Kubernetes concepts for PyTorch workloads

class KubernetesConfig:
    """Configuration for Kubernetes deployments"""
    
    def __init__(self):
        self.namespaces = {
            "development": "pytorch-dev",
            "staging": "pytorch-staging", 
            "production": "pytorch-prod"
        }
        
        self.resource_limits = {
            "training": {
                "cpu": "4",
                "memory": "8Gi",
                "nvidia.com/gpu": "1"
            },
            "inference": {
                "cpu": "2",
                "memory": "4Gi"
            },
            "jupyter": {
                "cpu": "2",
                "memory": "4Gi"
            }
        }
        
        self.storage_classes = {
            "fast": "ssd-fast",
            "standard": "standard",
            "archive": "hdd-archive"
        }

class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for PyTorch workloads"""
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
    
    def generate_namespace(self, environment: str) -> Dict[str, Any]:
        """Generate namespace manifest"""
        
        namespace_name = self.config.namespaces.get(environment, f"pytorch-{environment}")
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace_name,
                "labels": {
                    "app": "pytorch",
                    "environment": environment,
                    "created-by": "pytorch-ecosystem"
                }
            }
        }
        
        return manifest
    
    def generate_configmap(self, name: str, config_data: Dict[str, str],
                          namespace: str = "default") -> Dict[str, Any]:
        """Generate ConfigMap manifest"""
        
        manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "config"
                }
            },
            "data": config_data
        }
        
        return manifest
    
    def generate_secret(self, name: str, secret_data: Dict[str, str],
                       namespace: str = "default") -> Dict[str, Any]:
        """Generate Secret manifest"""
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "secret"
                }
            },
            "type": "Opaque",
            "data": secret_data  # Base64 encoded
        }
        
        return manifest
    
    def generate_pvc(self, name: str, size: str, storage_class: str = "standard",
                    namespace: str = "default") -> Dict[str, Any]:
        """Generate PersistentVolumeClaim manifest"""
        
        manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "storage"
                }
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": storage_class,
                "resources": {
                    "requests": {
                        "storage": size
                    }
                }
            }
        }
        
        return manifest
    
    def generate_training_job(self, name: str, image: str, 
                             command: List[str], args: List[str] = None,
                             namespace: str = "default",
                             gpu_count: int = 1) -> Dict[str, Any]:
        """Generate training Job manifest"""
        
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "training",
                    "job-name": name
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "pytorch",
                            "component": "training"
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "pytorch-trainer",
                            "image": image,
                            "command": command,
                            "args": args or [],
                            "resources": {
                                "limits": {
                                    "cpu": self.config.resource_limits["training"]["cpu"],
                                    "memory": self.config.resource_limits["training"]["memory"],
                                    "nvidia.com/gpu": str(gpu_count)
                                },
                                "requests": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                    "nvidia.com/gpu": str(gpu_count)
                                }
                            },
                            "env": [
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "TORCH_HOME", "value": "/workspace/.torch"}
                            ],
                            "volumeMounts": [
                                {
                                    "name": "data-volume",
                                    "mountPath": "/workspace/data"
                                },
                                {
                                    "name": "output-volume", 
                                    "mountPath": "/workspace/outputs"
                                }
                            ]
                        }],
                        "volumes": [
                            {
                                "name": "data-volume",
                                "persistentVolumeClaim": {
                                    "claimName": f"{name}-data-pvc"
                                }
                            },
                            {
                                "name": "output-volume",
                                "persistentVolumeClaim": {
                                    "claimName": f"{name}-output-pvc"
                                }
                            }
                        ]
                    }
                },
                "backoffLimit": 3
            }
        }
        
        return manifest
    
    def generate_inference_deployment(self, name: str, image: str,
                                     replicas: int = 3, namespace: str = "default") -> Dict[str, Any]:
        """Generate inference Deployment manifest"""
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "inference",
                    "deployment-name": name
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": "pytorch",
                        "component": "inference",
                        "deployment-name": name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "pytorch",
                            "component": "inference",
                            "deployment-name": name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "pytorch-inference",
                            "image": image,
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "limits": {
                                    "cpu": self.config.resource_limits["inference"]["cpu"],
                                    "memory": self.config.resource_limits["inference"]["memory"]
                                },
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "1Gi"
                                }
                            },
                            "env": [
                                {"name": "MODEL_PATH", "value": "/models/model.pth"},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "volumeMounts": [
                                {
                                    "name": "model-volume",
                                    "mountPath": "/models",
                                    "readOnly": True
                                }
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "volumes": [
                            {
                                "name": "model-volume",
                                "persistentVolumeClaim": {
                                    "claimName": f"{name}-model-pvc"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return manifest
    
    def generate_service(self, name: str, port: int = 8000,
                        target_port: int = 8000, service_type: str = "ClusterIP",
                        namespace: str = "default") -> Dict[str, Any]:
        """Generate Service manifest"""
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "service"
                }
            },
            "spec": {
                "type": service_type,
                "ports": [{
                    "port": port,
                    "targetPort": target_port,
                    "protocol": "TCP"
                }],
                "selector": {
                    "app": "pytorch",
                    "component": "inference"
                }
            }
        }
        
        if service_type == "LoadBalancer":
            manifest["spec"]["ports"][0]["nodePort"] = 30000
        
        return manifest
    
    def generate_ingress(self, name: str, host: str, service_name: str,
                        service_port: int = 8000, namespace: str = "default") -> Dict[str, Any]:
        """Generate Ingress manifest"""
        
        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "ingress"
                },
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [host],
                    "secretName": f"{name}-tls"
                }],
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": service_name,
                                    "port": {
                                        "number": service_port
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        return manifest
    
    def generate_hpa(self, name: str, deployment_name: str,
                    min_replicas: int = 2, max_replicas: int = 10,
                    cpu_threshold: int = 70, namespace: str = "default") -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest"""
        
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "autoscaler"
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": deployment_name
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": cpu_threshold
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        return manifest

class DistributedTrainingManifests:
    """Generate manifests for distributed PyTorch training"""
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
    
    def generate_pytorch_job(self, name: str, image: str, 
                           master_replicas: int = 1, worker_replicas: int = 3,
                           namespace: str = "default") -> Dict[str, Any]:
        """Generate PyTorchJob manifest (requires PyTorch Operator)"""
        
        manifest = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "distributed-training"
                }
            },
            "spec": {
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": master_replicas,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "pytorch",
                                    "role": "master"
                                }
                            },
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": image,
                                    "command": ["python", "/workspace/train_distributed.py"],
                                    "resources": {
                                        "limits": {
                                            "cpu": "4",
                                            "memory": "8Gi",
                                            "nvidia.com/gpu": "1"
                                        }
                                    },
                                    "env": [
                                        {"name": "PYTHONUNBUFFERED", "value": "1"}
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "data-volume",
                                            "mountPath": "/workspace/data"
                                        }
                                    ]
                                }],
                                "volumes": [
                                    {
                                        "name": "data-volume",
                                        "persistentVolumeClaim": {
                                            "claimName": f"{name}-data-pvc"
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    "Worker": {
                        "replicas": worker_replicas,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "pytorch",
                                    "role": "worker"
                                }
                            },
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": image,
                                    "command": ["python", "/workspace/train_distributed.py"],
                                    "resources": {
                                        "limits": {
                                            "cpu": "4",
                                            "memory": "8Gi",
                                            "nvidia.com/gpu": "1"
                                        }
                                    },
                                    "env": [
                                        {"name": "PYTHONUNBUFFERED", "value": "1"}
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "data-volume",
                                            "mountPath": "/workspace/data"
                                        }
                                    ]
                                }],
                                "volumes": [
                                    {
                                        "name": "data-volume",
                                        "persistentVolumeClaim": {
                                            "claimName": f"{name}-data-pvc"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        
        return manifest
    
    def generate_mpi_job(self, name: str, image: str,
                        worker_replicas: int = 4, namespace: str = "default") -> Dict[str, Any]:
        """Generate MPIJob manifest for Horovod training"""
        
        manifest = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "MPIJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "horovod-training"
                }
            },
            "spec": {
                "slotsPerWorker": 1,
                "runPolicy": {
                    "cleanPodPolicy": "Running"
                },
                "mpiReplicaSpecs": {
                    "Launcher": {
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "image": image,
                                    "name": "mpi-launcher",
                                    "command": [
                                        "mpirun",
                                        "-np", str(worker_replicas),
                                        "--hostfile", "/etc/mpi/hostfile",
                                        "python", "/workspace/train_horovod.py"
                                    ],
                                    "resources": {
                                        "limits": {
                                            "cpu": "2",
                                            "memory": "4Gi"
                                        }
                                    }
                                }]
                            }
                        }
                    },
                    "Worker": {
                        "replicas": worker_replicas,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "image": image,
                                    "name": "mpi-worker",
                                    "resources": {
                                        "limits": {
                                            "cpu": "4",
                                            "memory": "8Gi",
                                            "nvidia.com/gpu": "1"
                                        }
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "data-volume",
                                            "mountPath": "/workspace/data"
                                        }
                                    ]
                                }],
                                "volumes": [
                                    {
                                        "name": "data-volume",
                                        "persistentVolumeClaim": {
                                            "claimName": f"{name}-data-pvc"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        
        return manifest

class MonitoringManifests:
    """Generate monitoring and observability manifests"""
    
    def __init__(self):
        pass
    
    def generate_service_monitor(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Generate ServiceMonitor for Prometheus"""
        
        manifest = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "monitoring"
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app": "pytorch",
                        "component": "inference"
                    }
                },
                "endpoints": [{
                    "port": "metrics",
                    "interval": "30s",
                    "path": "/metrics"
                }]
            }
        }
        
        return manifest
    
    def generate_grafana_dashboard_configmap(self, name: str, 
                                           namespace: str = "default") -> Dict[str, Any]:
        """Generate ConfigMap with Grafana dashboard"""
        
        dashboard_json = {
            "dashboard": {
                "title": "PyTorch Inference Metrics",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(pytorch_requests_total[5m])"
                        }]
                    },
                    {
                        "title": "Response Time",
                        "type": "graph", 
                        "targets": [{
                            "expr": "pytorch_request_duration_seconds"
                        }]
                    },
                    {
                        "title": "Model Inference Time",
                        "type": "graph",
                        "targets": [{
                            "expr": "pytorch_inference_duration_seconds"
                        }]
                    }
                ]
            }
        }
        
        manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "pytorch",
                    "component": "monitoring",
                    "grafana_dashboard": "1"
                }
            },
            "data": {
                "pytorch-dashboard.json": json.dumps(dashboard_json, indent=2)
            }
        }
        
        return manifest

class KubernetesDeploymentManager:
    """Manage Kubernetes deployments for PyTorch workloads"""
    
    def __init__(self, project_name: str = "pytorch-k8s"):
        self.project_name = project_name
        self.config = KubernetesConfig()
        self.manifest_generator = KubernetesManifestGenerator(self.config)
        self.distributed_manifests = DistributedTrainingManifests(self.config)
        self.monitoring_manifests = MonitoringManifests()
    
    def generate_complete_deployment(self, project_path: Path,
                                   environment: str = "development") -> List[str]:
        """Generate complete Kubernetes deployment"""
        
        k8s_dir = project_path / "k8s" / environment
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        manifests_created = []
        
        # 1. Namespace
        namespace_manifest = self.manifest_generator.generate_namespace(environment)
        self._save_manifest(k8s_dir / "01-namespace.yaml", namespace_manifest)
        manifests_created.append("01-namespace.yaml")
        
        # 2. ConfigMaps
        config_data = {
            "model_config.yaml": """
model:
  type: "classification"
  num_classes: 10
  hidden_size: 128

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
""",
            "logging.conf": """
[loggers]
keys=root

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""
        }
        
        configmap_manifest = self.manifest_generator.generate_configmap(
            "pytorch-config", config_data, self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "02-configmap.yaml", configmap_manifest)
        manifests_created.append("02-configmap.yaml")
        
        # 3. Secrets (example)
        secret_data = {
            "wandb-api-key": "d2FuZGJfa2V5X2hlcmU=",  # base64 encoded
            "aws-access-key": "YXdzX2tleV9oZXJl"      # base64 encoded
        }
        
        secret_manifest = self.manifest_generator.generate_secret(
            "pytorch-secrets", secret_data, self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "03-secrets.yaml", secret_manifest)
        manifests_created.append("03-secrets.yaml")
        
        # 4. PVCs
        pvcs = [
            ("data-pvc", "10Gi", "standard"),
            ("model-pvc", "5Gi", "fast"),
            ("output-pvc", "20Gi", "standard")
        ]
        
        for pvc_name, size, storage_class in pvcs:
            pvc_manifest = self.manifest_generator.generate_pvc(
                pvc_name, size, storage_class, self.config.namespaces[environment]
            )
            self._save_manifest(k8s_dir / f"04-{pvc_name}.yaml", pvc_manifest)
            manifests_created.append(f"04-{pvc_name}.yaml")
        
        # 5. Training Job
        training_job = self.manifest_generator.generate_training_job(
            "pytorch-training",
            "pytorch-app:training",
            ["python", "/workspace/train.py"],
            ["--epochs", "50", "--batch-size", "64"],
            self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "05-training-job.yaml", training_job)
        manifests_created.append("05-training-job.yaml")
        
        # 6. Inference Deployment
        inference_deployment = self.manifest_generator.generate_inference_deployment(
            "pytorch-inference",
            "pytorch-app:inference",
            replicas=3,
            namespace=self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "06-inference-deployment.yaml", inference_deployment)
        manifests_created.append("06-inference-deployment.yaml")
        
        # 7. Service
        service_manifest = self.manifest_generator.generate_service(
            "pytorch-inference-service",
            port=8000,
            namespace=self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "07-service.yaml", service_manifest)
        manifests_created.append("07-service.yaml")
        
        # 8. Ingress
        ingress_manifest = self.manifest_generator.generate_ingress(
            "pytorch-inference-ingress",
            f"pytorch-{environment}.example.com",
            "pytorch-inference-service",
            namespace=self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "08-ingress.yaml", ingress_manifest)
        manifests_created.append("08-ingress.yaml")
        
        # 9. HPA
        hpa_manifest = self.manifest_generator.generate_hpa(
            "pytorch-inference-hpa",
            "pytorch-inference",
            min_replicas=2,
            max_replicas=10,
            namespace=self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "09-hpa.yaml", hpa_manifest)
        manifests_created.append("09-hpa.yaml")
        
        # 10. Monitoring
        service_monitor = self.monitoring_manifests.generate_service_monitor(
            "pytorch-service-monitor",
            self.config.namespaces[environment]
        )
        self._save_manifest(k8s_dir / "10-service-monitor.yaml", service_monitor)
        manifests_created.append("10-service-monitor.yaml")
        
        print(f"✓ Generated {len(manifests_created)} Kubernetes manifests for {environment}")
        return manifests_created
    
    def generate_distributed_training_manifests(self, project_path: Path) -> List[str]:
        """Generate distributed training manifests"""
        
        dist_dir = project_path / "k8s" / "distributed"
        dist_dir.mkdir(parents=True, exist_ok=True)
        
        manifests_created = []
        
        # PyTorchJob
        pytorch_job = self.distributed_manifests.generate_pytorch_job(
            "pytorch-distributed-training",
            "pytorch-app:distributed",
            master_replicas=1,
            worker_replicas=4
        )
        self._save_manifest(dist_dir / "pytorch-job.yaml", pytorch_job)
        manifests_created.append("pytorch-job.yaml")
        
        # MPIJob
        mpi_job = self.distributed_manifests.generate_mpi_job(
            "pytorch-horovod-training",
            "pytorch-app:horovod",
            worker_replicas=4
        )
        self._save_manifest(dist_dir / "mpi-job.yaml", mpi_job)
        manifests_created.append("mpi-job.yaml")
        
        print(f"✓ Generated {len(manifests_created)} distributed training manifests")
        return manifests_created
    
    def generate_kustomization_files(self, project_path: Path):
        """Generate Kustomization files for environment management"""
        
        environments = ["development", "staging", "production"]
        
        for env in environments:
            env_dir = project_path / "k8s" / env
            env_dir.mkdir(parents=True, exist_ok=True)
            
            kustomization = {
                "apiVersion": "kustomize.config.k8s.io/v1beta1",
                "kind": "Kustomization",
                "resources": [
                    "01-namespace.yaml",
                    "02-configmap.yaml",
                    "03-secrets.yaml",
                    "04-data-pvc.yaml",
                    "04-model-pvc.yaml",
                    "04-output-pvc.yaml",
                    "05-training-job.yaml",
                    "06-inference-deployment.yaml",
                    "07-service.yaml",
                    "08-ingress.yaml",
                    "09-hpa.yaml",
                    "10-service-monitor.yaml"
                ],
                "namePrefix": f"{env}-",
                "namespace": self.config.namespaces[env],
                "images": [
                    {
                        "name": "pytorch-app",
                        "newTag": f"{env}-latest"
                    }
                ]
            }
            
            # Environment-specific patches
            if env == "production":
                kustomization["patchesStrategicMerge"] = [
                    "patches/production-resources.yaml"
                ]
            
            self._save_manifest(env_dir / "kustomization.yaml", kustomization)
        
        print("✓ Generated Kustomization files for all environments")
    
    def generate_deployment_scripts(self, project_path: Path):
        """Generate deployment and management scripts"""
        
        scripts_dir = project_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Deploy script
        deploy_script = '''#!/bin/bash
set -e

ENVIRONMENT=${1:-"development"}
ACTION=${2:-"apply"}

echo "Deploying PyTorch application to $ENVIRONMENT environment..."

case $ACTION in
    "apply")
        kubectl apply -k k8s/$ENVIRONMENT/
        ;;
    "delete")
        kubectl delete -k k8s/$ENVIRONMENT/
        ;;
    "diff")
        kubectl diff -k k8s/$ENVIRONMENT/
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Available actions: apply, delete, diff"
        exit 1
        ;;
esac

echo "Operation completed!"
'''
        
        with open(scripts_dir / "deploy.sh", 'w') as f:
            f.write(deploy_script)
        
        # Monitoring script
        monitor_script = '''#!/bin/bash
set -e

NAMESPACE=${1:-"pytorch-dev"}

echo "Monitoring PyTorch workloads in namespace: $NAMESPACE"

echo "=== Pods ==="
kubectl get pods -n $NAMESPACE -l app=pytorch

echo ""
echo "=== Services ==="
kubectl get services -n $NAMESPACE -l app=pytorch

echo ""
echo "=== Deployments ==="
kubectl get deployments -n $NAMESPACE -l app=pytorch

echo ""
echo "=== Jobs ==="
kubectl get jobs -n $NAMESPACE -l app=pytorch

echo ""
echo "=== HPA ==="
kubectl get hpa -n $NAMESPACE -l app=pytorch

echo ""
echo "=== Ingress ==="
kubectl get ingress -n $NAMESPACE -l app=pytorch
'''
        
        with open(scripts_dir / "monitor.sh", 'w') as f:
            f.write(monitor_script)
        
        # Logs script
        logs_script = '''#!/bin/bash
set -e

NAMESPACE=${1:-"pytorch-dev"}
COMPONENT=${2:-"inference"}

echo "Getting logs for $COMPONENT in namespace: $NAMESPACE"

# Get pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=pytorch,component=$COMPONENT -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "No pods found for component: $COMPONENT"
    exit 1
fi

echo "Following logs for pod: $POD_NAME"
kubectl logs -n $NAMESPACE -f $POD_NAME
'''
        
        with open(scripts_dir / "logs.sh", 'w') as f:
            f.write(logs_script)
        
        # Scale script
        scale_script = '''#!/bin/bash
set -e

NAMESPACE=${1:-"pytorch-dev"}
DEPLOYMENT=${2:-"pytorch-inference"}
REPLICAS=${3:-3}

echo "Scaling $DEPLOYMENT to $REPLICAS replicas in namespace: $NAMESPACE"

kubectl scale deployment $DEPLOYMENT -n $NAMESPACE --replicas=$REPLICAS

echo "Waiting for rollout to complete..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

echo "Scaling completed!"
'''
        
        with open(scripts_dir / "scale.sh", 'w') as f:
            f.write(scale_script)
        
        # Make scripts executable
        for script in ["deploy.sh", "monitor.sh", "logs.sh", "scale.sh"]:
            os.chmod(scripts_dir / script, 0o755)
        
        print("✓ Generated Kubernetes management scripts")
    
    def _save_manifest(self, path: Path, manifest: Dict[str, Any]):
        """Save manifest to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

class KubernetesUtilities:
    """Utility functions for Kubernetes operations"""
    
    @staticmethod
    def check_kubectl_availability() -> bool:
        """Check if kubectl is available"""
        try:
            import subprocess
            result = subprocess.run(["kubectl", "version", "--client"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def get_cluster_info() -> Dict[str, Any]:
        """Get cluster information"""
        if not KubernetesUtilities.check_kubectl_availability():
            return {"available": False}
        
        try:
            import subprocess
            result = subprocess.run(["kubectl", "cluster-info"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return {
                    "available": True,
                    "cluster_info": result.stdout.strip()
                }
        except Exception:
            pass
        
        return {"available": True, "info": "limited"}
    
    @staticmethod
    def kubernetes_best_practices() -> List[str]:
        """Kubernetes best practices for ML workloads"""
        
        practices = [
            "Use resource requests and limits for all containers",
            "Implement proper health checks (liveness, readiness)",
            "Use namespaces to organize workloads by environment",
            "Apply the principle of least privilege with RBAC",
            "Use secrets for sensitive data, not ConfigMaps",
            "Implement horizontal pod autoscaling for inference",
            "Use persistent volumes for model artifacts and data",
            "Tag images with specific versions, not 'latest'",
            "Use init containers for setup tasks",
            "Implement proper logging and monitoring",
            "Use node selectors or affinity for GPU workloads",
            "Apply network policies for security",
            "Use StatefulSets for distributed training coordination",
            "Implement graceful shutdown handling",
            "Use pod disruption budgets for availability"
        ]
        
        return practices

if __name__ == "__main__":
    print("Kubernetes PyTorch Scaling")
    print("=" * 29)
    
    print("\n1. Kubernetes Environment Check")
    print("-" * 33)
    
    kubectl_available = KubernetesUtilities.check_kubectl_availability()
    
    if kubectl_available:
        print("✓ kubectl is available")
        cluster_info = KubernetesUtilities.get_cluster_info()
        
        if cluster_info.get("cluster_info"):
            print("✓ Connected to Kubernetes cluster")
        else:
            print("⚠ kubectl available but no cluster connection")
    else:
        print("✗ kubectl not available")
        print("  Install kubectl to manage Kubernetes deployments")
    
    print("\n2. Kubernetes Deployment Setup")
    print("-" * 33)
    
    # Create deployment manager
    deployment_manager = KubernetesDeploymentManager("pytorch-k8s-demo")
    
    # Create project structure
    project_path = Path("./pytorch-k8s-demo")
    project_path.mkdir(exist_ok=True)
    
    print(f"✓ Created project directory: {project_path}")
    
    print("\n3. Manifest Generation")
    print("-" * 24)
    
    # Generate manifests for different environments
    environments = ["development", "staging", "production"]
    
    for env in environments:
        manifests = deployment_manager.generate_complete_deployment(project_path, env)
        print(f"  {env}: {len(manifests)} manifests")
    
    # Generate distributed training manifests
    dist_manifests = deployment_manager.generate_distributed_training_manifests(project_path)
    print(f"  Distributed training: {len(dist_manifests)} manifests")
    
    print("\n4. Kustomization Setup")
    print("-" * 23)
    
    deployment_manager.generate_kustomization_files(project_path)
    
    print("\n5. Management Scripts")
    print("-" * 22)
    
    deployment_manager.generate_deployment_scripts(project_path)
    
    print("\n6. Kubernetes Best Practices")
    print("-" * 31)
    
    best_practices = KubernetesUtilities.kubernetes_best_practices()
    
    print("Kubernetes Best Practices for ML:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Deployment Commands")
    print("-" * 23)
    
    commands = {
        "Deploy to development": "./scripts/deploy.sh development apply",
        "Deploy to production": "./scripts/deploy.sh production apply",
        "Monitor workloads": "./scripts/monitor.sh pytorch-prod",
        "View logs": "./scripts/logs.sh pytorch-prod inference",
        "Scale deployment": "./scripts/scale.sh pytorch-prod pytorch-inference 5",
        "Check pod status": "kubectl get pods -n pytorch-prod -l app=pytorch",
        "Port forward service": "kubectl port-forward svc/pytorch-inference-service 8000:8000 -n pytorch-prod"
    }
    
    print("Common Kubernetes Commands:")
    for command, example in commands.items():
        print(f"  {command}: {example}")
    
    print("\n8. Scaling Strategies")
    print("-" * 21)
    
    scaling_strategies = {
        "Horizontal Pod Autoscaling": "Scale replicas based on CPU/memory usage",
        "Vertical Pod Autoscaling": "Adjust resource requests/limits automatically",
        "Cluster Autoscaling": "Scale cluster nodes based on pending pods",
        "Manual Scaling": "Explicitly set replica count for predictable loads",
        "Custom Metrics Scaling": "Scale based on custom metrics (queue length, etc.)",
        "Scheduled Scaling": "Scale based on time patterns using CronJobs"
    }
    
    print("Kubernetes Scaling Strategies:")
    for strategy, description in scaling_strategies.items():
        print(f"  {strategy}: {description}")
    
    print("\n9. Resource Management")
    print("-" * 23)
    
    resource_tips = [
        "Set appropriate CPU and memory requests/limits",
        "Use GPU node selectors for training workloads",
        "Implement pod priority classes for workload prioritization",
        "Use resource quotas to limit namespace consumption",
        "Monitor resource utilization with metrics server",
        "Use limit ranges to enforce container resource constraints",
        "Configure quality of service classes (Guaranteed, Burstable, BestEffort)",
        "Use pod disruption budgets to maintain availability during updates"
    ]
    
    print("Resource Management Tips:")
    for i, tip in enumerate(resource_tips, 1):
        print(f"{i}. {tip}")
    
    print("\n10. Production Considerations")
    print("-" * 32)
    
    production_considerations = [
        "Multi-zone deployment for high availability",
        "Backup and disaster recovery procedures",
        "Security scanning and compliance",
        "Network policies for microsegmentation",
        "Service mesh for advanced traffic management",
        "GitOps for declarative deployment management",
        "Cost optimization and resource governance",
        "Performance monitoring and alerting",
        "Capacity planning and forecasting",
        "Incident response and troubleshooting procedures"
    ]
    
    print("Production Considerations:")
    for consideration in production_considerations:
        print(f"  - {consideration}")
    
    print("\nKubernetes PyTorch scaling demonstration completed!")
    print("Key components covered:")
    print("  - Complete Kubernetes manifest generation")
    print("  - Multi-environment deployment with Kustomize")
    print("  - Distributed training with PyTorchJob and MPIJob")
    print("  - Autoscaling and resource management")
    print("  - Monitoring and observability setup")
    print("  - Production deployment best practices")
    
    print("\nKubernetes enables:")
    print("  - Scalable and resilient ML model deployment")
    print("  - Efficient resource utilization and management")
    print("  - Automated scaling based on demand")
    print("  - Multi-environment deployment consistency")
    print("  - Enterprise-grade security and compliance")