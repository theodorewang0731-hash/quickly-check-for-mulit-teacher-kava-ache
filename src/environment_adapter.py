"""
环境自适应模块
自动检测并适配不同的运行环境（本地、HPC、云平台）
确保代码在任何环境下都能正确运行
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml
import warnings

import torch
import transformers


class EnvironmentAdapter:
    """环境自适应器 - 自动检测并配置运行环境"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化环境适配器
        
        Args:
            config_path: 配置文件路径，默认为 configs/environment_config.yaml
        """
        self.project_root = Path(__file__).parent.parent
        
        # 加载配置
        if config_path is None:
            config_path = self.project_root / "configs" / "environment_config.yaml"
        self.config = self._load_config(config_path)
        
        # 检测环境
        self.env_info = self._detect_environment()
        
        # 配置硬件
        self.hardware_config = self._configure_hardware()
        
        # 配置路径
        self.paths = self._configure_paths()
        
        # 检测依赖
        self.dependencies = self._detect_dependencies()
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            warnings.warn(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'environment_type': 'auto',
            'hardware': {
                'gpu_detection': {'auto_detect': True},
                'precision': {'auto_detect': True}
            },
            'model_dimensions': {'auto_detect': True},
            'paths': {'auto_detect': True},
            'dependencies': {'auto_detect': True}
        }
    
    def _detect_environment(self) -> Dict[str, Any]:
        """
        检测运行环境类型
        
        Returns:
            包含环境信息的字典
        """
        env_info = {
            'type': 'unknown',
            'platform': platform.system(),
            'python_version': sys.version,
            'hostname': platform.node(),
            'cpu_count': os.cpu_count(),
        }
        
        # 检测 HPC 环境特征
        hpc_indicators = [
            'SLURM_JOB_ID',      # SLURM
            'PBS_JOBID',         # PBS
            'SGE_TASK_ID',       # SGE
            'LSB_JOBID',         # LSF
            'COBALT_JOBID',      # Cobalt
        ]
        
        if any(indicator in os.environ for indicator in hpc_indicators):
            env_info['type'] = 'hpc'
            env_info['scheduler'] = self._detect_scheduler()
        
        # 检测云平台
        elif 'KUBERNETES_SERVICE_HOST' in os.environ:
            env_info['type'] = 'cloud'
            env_info['platform_type'] = 'kubernetes'
        
        # 默认为本地环境
        else:
            env_info['type'] = 'local'
        
        return env_info
    
    def _detect_scheduler(self) -> Optional[str]:
        """检测 HPC 作业调度器"""
        if 'SLURM_JOB_ID' in os.environ:
            return 'slurm'
        elif 'PBS_JOBID' in os.environ:
            return 'pbs'
        elif 'SGE_TASK_ID' in os.environ:
            return 'sge'
        elif 'LSB_JOBID' in os.environ:
            return 'lsf'
        return None
    
    def _configure_hardware(self) -> Dict[str, Any]:
        """
        配置硬件（GPU、精度、内存等）
        
        Returns:
            硬件配置字典
        """
        hw_config = {
            'device': 'cpu',
            'device_name': 'CPU',
            'precision': 'fp32',
            'num_gpus': 0,
            'memory_gb': 0,
            'supports_bf16': False,
            'supports_fp16': False,
        }
        
        # 检测 GPU
        if torch.cuda.is_available():
            hw_config['device'] = 'cuda'
            hw_config['num_gpus'] = torch.cuda.device_count()
            hw_config['device_name'] = torch.cuda.get_device_name(0)
            hw_config['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # 检测精度支持
            hw_config['supports_fp16'] = True
            hw_config['supports_bf16'] = torch.cuda.is_bf16_supported()
            
            # 自动选择最佳精度
            if hw_config['supports_bf16']:
                hw_config['precision'] = 'bf16'
            elif hw_config['supports_fp16']:
                hw_config['precision'] = 'fp16'
            
            # 配置显存管理
            if self.config['hardware']['gpu_detection'].get('allow_growth', True):
                torch.cuda.empty_cache()
        
        elif torch.backends.mps.is_available():
            hw_config['device'] = 'mps'
            hw_config['device_name'] = 'Apple Silicon'
            hw_config['supports_fp16'] = True
        
        return hw_config
    
    def _configure_paths(self) -> Dict[str, Path]:
        """
        配置路径（模型、数据、输出等）
        支持环境变量和相对路径
        
        Returns:
            路径配置字典
        """
        paths = {}
        path_config = self.config.get('paths', {})
        defaults = path_config.get('defaults', {})
        env_vars = path_config.get('env_vars', {})
        
        # 为每个路径类型配置
        for path_type, default_path in defaults.items():
            # 1. 尝试从环境变量读取
            env_var = env_vars.get(path_type)
            if env_var and env_var in os.environ:
                paths[path_type] = Path(os.environ[env_var])
            
            # 2. HPC 环境特殊处理
            elif self.env_info['type'] == 'hpc':
                username = os.environ.get('USER', 'user')
                hpc_patterns = path_config.get('hpc_patterns', [])
                
                # 尝试每个 HPC 路径模式
                for pattern in hpc_patterns:
                    hpc_path = Path(pattern.format(username=username)) / path_type
                    if hpc_path.exists() or pattern.startswith('/scratch'):
                        paths[path_type] = hpc_path
                        break
                else:
                    # 回退到默认路径
                    paths[path_type] = self.project_root / default_path
            
            # 3. 使用默认相对路径
            else:
                paths[path_type] = self.project_root / default_path
            
            # 确保目录存在
            paths[path_type].mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def _detect_dependencies(self) -> Dict[str, Any]:
        """
        检测可用的依赖库
        
        Returns:
            依赖信息字典
        """
        deps = {
            'torch': {
                'available': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
            },
            'transformers': {
                'available': True,
                'version': transformers.__version__,
            }
        }
        
        # 检测可选依赖
        optional_deps = {
            'accelerate': 'accelerate',
            'bitsandbytes': 'bitsandbytes',
            'flash_attn': 'flash_attn',
            'deepspeed': 'deepspeed',
            'wandb': 'wandb',
        }
        
        for dep_name, import_name in optional_deps.items():
            try:
                module = __import__(import_name)
                deps[dep_name] = {
                    'available': True,
                    'version': getattr(module, '__version__', 'unknown')
                }
            except ImportError:
                deps[dep_name] = {'available': False}
        
        return deps
    
    def get_device(self) -> torch.device:
        """
        获取推荐的设备
        
        Returns:
            torch.device 对象
        """
        return torch.device(self.hardware_config['device'])
    
    def get_dtype(self) -> torch.dtype:
        """
        获取推荐的数据类型
        
        Returns:
            torch.dtype 对象
        """
        precision = self.hardware_config['precision']
        dtype_map = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }
        return dtype_map.get(precision, torch.float32)
    
    def get_optimal_batch_size(self, base_size: int = 2) -> Tuple[int, int]:
        """
        根据硬件自动计算最优 batch size 和梯度累积步数
        
        Args:
            base_size: 基础 batch size
            
        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if not self.config['training'].get('auto_tune', True):
            return base_size, 1
        
        memory_gb = self.hardware_config['memory_gb']
        target_batch = self.config['training']['gradient_accumulation'].get('target_batch_size', 32)
        
        # 根据显存估算合适的 batch size
        if memory_gb >= 40:  # A100 40GB+
            batch_size = 8
        elif memory_gb >= 24:  # RTX 4090, A10
            batch_size = 4
        elif memory_gb >= 16:  # RTX 4080
            batch_size = 2
        elif memory_gb >= 8:   # RTX 4070
            batch_size = 2
        else:
            batch_size = 1
        
        # 计算梯度累积步数
        grad_accum = max(1, target_batch // batch_size)
        
        return batch_size, grad_accum
    
    def detect_kv_dimensions(self, model, max_length: int = 32) -> int:
        """
        运行时动态检测模型 KV Cache 的实际维度
        
        Args:
            model: 加载的模型
            max_length: 测试序列长度
            
        Returns:
            实际 KV 维度
        """
        device = self.get_device()
        
        # 创建测试输入
        test_input = torch.randint(0, 1000, (1, max_length)).to(device)
        
        # 前向传播获取 KV cache
        with torch.no_grad():
            outputs = model(test_input, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # 计算总维度（所有层的维度之和）
            total_dim = 0
            for layer_kv in past_key_values:
                k, v = layer_kv
                # k shape: [B, num_heads, seq_len, head_dim]
                B, H, T, D_h = k.shape
                layer_dim = H * D_h
                total_dim += layer_dim
        
        return total_dim
    
    def print_environment_info(self):
        """打印环境信息"""
        print("\n" + "="*70)
        print("[Environment Detection Report]")
        print("="*70)
        
        # 环境类型
        print(f"\n[Environment Type]: {self.env_info['type'].upper()}")
        print(f"   Platform: {self.env_info['platform']}")
        print(f"   Hostname: {self.env_info['hostname']}")
        print(f"   CPU Cores: {self.env_info['cpu_count']}")
        
        # 硬件配置
        print(f"\n[Hardware Configuration]:")
        print(f"   Device: {self.hardware_config['device'].upper()}")
        print(f"   Name: {self.hardware_config['device_name']}")
        if self.hardware_config['num_gpus'] > 0:
            print(f"   GPUs: {self.hardware_config['num_gpus']}")
            print(f"   Memory: {self.hardware_config['memory_gb']:.1f} GB")
        print(f"   Precision: {self.hardware_config['precision'].upper()}")
        print(f"   BF16 Support: {'YES' if self.hardware_config['supports_bf16'] else 'NO'}")
        
        # 路径配置
        print(f"\n[Path Configuration]:")
        for path_type, path in self.paths.items():
            print(f"   {path_type}: {path}")
        
        # 依赖检测
        print(f"\n[Dependencies]:")
        for dep_name, dep_info in self.dependencies.items():
            if dep_info['available']:
                version = dep_info.get('version', 'unknown')
                print(f"   [OK] {dep_name} ({version})")
            else:
                print(f"   [X] {dep_name} (not installed)")
        
        print("="*70 + "\n")
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        获取完整的训练配置
        
        Returns:
            训练配置字典
        """
        batch_size, grad_accum = self.get_optimal_batch_size()
        
        return {
            'device': self.get_device(),
            'dtype': self.get_dtype(),
            'batch_size': batch_size,
            'gradient_accumulation_steps': grad_accum,
            'effective_batch_size': batch_size * grad_accum,
            'paths': self.paths,
            'mixed_precision': self.hardware_config['precision'],
        }


# ====================================================================
# 便捷函数
# ====================================================================

def create_environment_adapter(config_path: Optional[str] = None) -> EnvironmentAdapter:
    """
    创建环境适配器（便捷函数）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        EnvironmentAdapter 实例
    """
    adapter = EnvironmentAdapter(config_path)
    adapter.print_environment_info()
    return adapter


if __name__ == "__main__":
    # 测试环境检测
    adapter = create_environment_adapter()
    
    print("\n🎯 Recommended Training Configuration:")
    config = adapter.get_training_config()
    for key, value in config.items():
        if key != 'paths':
            print(f"   {key}: {value}")
