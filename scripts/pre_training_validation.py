#!/usr/bin/env python3
"""
KaVa 多教师蒸馏项目 - 训练前完整验证脚本

根据 KAVA 项目实施经验，检查以下关键问题：
1. 路径空格处理
2. Windows 换行符
3. 参数匹配性
4. GPU 格式配置
5. 资源配置合理性
6. 环境完整性
7. 模型可访问性
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util
import re
import json

# ANSI 颜色
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"{BLUE}{title}{RESET}")
    print('='*60)


def print_check(item, status, message=""):
    symbol = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    print(f"{symbol} {item}")
    if message:
        print(f"  {YELLOW}{message}{RESET}")


class KaVaPreTrainingValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.errors = []
        self.warnings = []
        
    def check_all(self):
        """运行所有检查"""
        print(f"{BLUE}KaVa 多教师蒸馏项目 - 训练前验证{RESET}")
        print(f"项目路径: {self.project_root}\n")
        
        # 按顺序执行检查
        checks = [
            ("检查项目结构", self.check_project_structure),
            ("检查 Python 环境", self.check_python_environment),
            ("检查核心代码模块", self.check_core_modules),
            ("检查 SLURM 脚本", self.check_slurm_scripts),
            ("检查路径空格处理", self.check_path_quoting),
            ("检查换行符格式", self.check_line_endings),
            ("检查 GPU 配置", self.check_gpu_configuration),
            ("检查资源配置", self.check_resource_configuration),
            ("检查模型路径", self.check_model_paths),
            ("检查数据路径", self.check_data_paths),
        ]
        
        for title, check_func in checks:
            print_header(title)
            try:
                check_func()
            except Exception as e:
                self.errors.append(f"{title}: {str(e)}")
                print_check(title, False, f"异常: {str(e)}")
        
        # 最终总结
        self.print_summary()
        
    def check_project_structure(self):
        """检查项目文件结构"""
        required_files = [
            "requirements.txt",
            "scripts/run_multi_seed_experiments.sh",
            "scripts/setup_hpc_environment.sh",
            "experiments/train_multi_teacher_kv.py",
            "align/tokenizer_align.py",
            "teacher/router_proto.py",
            "data/data_split_controller.py",
        ]
        
        missing = []
        for file in required_files:
            path = self.project_root / file
            if path.exists():
                print_check(file, True)
            else:
                print_check(file, False, "文件不存在")
                missing.append(file)
        
        if missing:
            self.errors.append(f"缺少 {len(missing)} 个关键文件")
            
    def check_python_environment(self):
        """检查 Python 环境和依赖"""
        # 检查 Python 版本
        py_version = sys.version_info
        if py_version.major == 3 and py_version.minor >= 10:
            print_check(f"Python {py_version.major}.{py_version.minor}", True)
        else:
            print_check(f"Python {py_version.major}.{py_version.minor}", False, "需要 Python 3.10+")
            self.errors.append("Python 版本不符合要求")
        
        # 检查关键包
        required_packages = [
            "torch",
            "transformers",
            "accelerate",
            "datasets",
            "numpy",
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
        ]
        
        for pkg in required_packages:
            try:
                mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
                version = getattr(mod, "__version__", "unknown")
                print_check(f"{pkg:15s} {version}", True)
            except ImportError:
                print_check(f"{pkg:15s}", False, "未安装")
                self.errors.append(f"{pkg} 未安装")
                
        # 检查 PyTorch CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print_check(f"PyTorch CUDA {torch.version.cuda}", True, f"检测到 {torch.cuda.device_count()} 个 GPU")
            else:
                print_check("PyTorch CUDA", False, "CUDA 不可用（登录节点正常）")
                self.warnings.append("登录节点无 CUDA（需要在计算节点验证）")
        except:
            self.errors.append("PyTorch 检查失败")
            
    def check_core_modules(self):
        """检查核心代码模块的导入"""
        sys.path.insert(0, str(self.project_root))
        
        modules_to_check = [
            ("align.tokenizer_align", "TokenizerAligner"),
            ("align.layer_map", "LayerMapper"),
            ("teacher.router_proto", "RouterProto"),
            ("data.data_split_controller", "DataSplitController"),
            ("utils.training_budget_controller", "TrainingBudgetController"),
            ("utils.statistical_significance", "MultiSeedAggregator"),
            ("utils.learning_curve_tracker", "LearningCurveTracker"),
        ]
        
        for module_path, class_name in modules_to_check:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    print_check(f"{module_path}.{class_name}", True)
                else:
                    print_check(f"{module_path}.{class_name}", False, f"类 {class_name} 不存在")
                    self.warnings.append(f"{module_path}.{class_name} 缺失")
            except Exception as e:
                print_check(f"{module_path}", False, str(e))
                self.errors.append(f"{module_path} 导入失败: {str(e)}")
                
    def check_slurm_scripts(self):
        """检查 SLURM 脚本的基本语法"""
        slurm_scripts = [
            "scripts/run_multi_seed_experiments.sh",
            "scripts/run_ablation_studies.sh",
            "scripts/setup_hpc_environment.sh",
            "scripts/check_gpu_node.sh",
        ]
        
        for script_path in slurm_scripts:
            full_path = self.project_root / script_path
            if not full_path.exists():
                print_check(script_path, False, "文件不存在")
                continue
                
            # 检查 shebang
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!/bin/bash') or first_line.startswith('#!/bin/sh'):
                    print_check(f"{script_path} (shebang)", True)
                else:
                    print_check(f"{script_path} (shebang)", False, f"缺少 shebang: {first_line}")
                    self.warnings.append(f"{script_path} 缺少正确的 shebang")
                    
    def check_path_quoting(self):
        """检查脚本中的路径变量是否正确加引号（问题 5: 路径空格处理）"""
        slurm_scripts = list(self.project_root.glob("scripts/*.sh"))
        
        # 检测未加引号的路径变量
        unquoted_pattern = re.compile(r'\bcd\s+\$\w+|\bcd\s+\$\{\w+\}')
        quoted_pattern = re.compile(r'\bcd\s+"?\$')
        
        issues = []
        for script in slurm_scripts:
            with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # 跳过注释行
                    if line.strip().startswith('#'):
                        continue
                        
                    # 检测 cd $VARIABLE（未加引号）
                    if 'cd ' in line and '$' in line:
                        if unquoted_pattern.search(line) and '"$' not in line and "'$" not in line:
                            issues.append((script.name, i, line.strip()))
        
        if issues:
            print_check("路径变量引号检查", False)
            for script_name, line_num, line in issues:
                print(f"  {YELLOW}{script_name}:{line_num}: {line}{RESET}")
            self.warnings.append(f"发现 {len(issues)} 处未加引号的路径变量")
        else:
            print_check("路径变量引号检查", True, "所有路径变量已正确加引号")
            
    def check_line_endings(self):
        """检查脚本文件的换行符格式（问题 10: Windows 换行符）"""
        shell_scripts = list(self.project_root.glob("scripts/*.sh"))
        
        crlf_files = []
        for script in shell_scripts:
            with open(script, 'rb') as f:
                content = f.read()
                if b'\r\n' in content:
                    crlf_files.append(script.name)
        
        if crlf_files:
            print_check("换行符检查", False)
            for fname in crlf_files:
                print(f"  {YELLOW}{fname}: 包含 Windows 换行符 (CRLF){RESET}")
            self.errors.append(f"{len(crlf_files)} 个脚本包含 Windows 换行符")
            print(f"\n  {BLUE}修复命令:{RESET}")
            print(f"    sed -i 's/\\r$//' scripts/*.sh")
            print(f"    # 或")
            print(f"    dos2unix scripts/*.sh")
        else:
            print_check("换行符检查", True, "所有脚本使用 Unix 换行符 (LF)")
            
    def check_gpu_configuration(self):
        """检查 GPU 配置格式（问题 8: GPU 格式问题）"""
        slurm_files = list(self.project_root.glob("scripts/*.sh"))
        
        # 检查 #SBATCH --gres 行
        gres_configs = []
        for script in slurm_files:
            with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(r'#SBATCH\s+--gres=(\S+)', content)
                if matches:
                    gres_configs.append((script.name, matches))
        
        if gres_configs:
            print("发现的 GPU 配置:")
            for script_name, configs in gres_configs:
                for config in configs:
                    print(f"  {script_name}: --gres={config}")
                    
                    # 检查是否是完整格式
                    if ':' in config and config.count(':') >= 2:
                        print_check(f"  格式检查", True, "使用完整 GPU 格式")
                    else:
                        print_check(f"  格式检查", False, "可能需要完整格式（如 gpu:a100-sxm4-80gb:1）")
                        self.warnings.append(f"{script_name}: GPU 格式可能不匹配")
        else:
            print_check("GPU 配置", False, "未找到 #SBATCH --gres 配置")
            self.warnings.append("SLURM 脚本中没有 GPU 配置")
            
    def check_resource_configuration(self):
        """检查资源配置合理性（问题 7: 资源配置错误）"""
        slurm_files = list(self.project_root.glob("scripts/*.sh"))
        
        for script in slurm_files:
            with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 提取资源配置
                cpus = re.search(r'#SBATCH\s+--cpus-per-task=(\d+)', content)
                mem = re.search(r'#SBATCH\s+--mem=(\d+)G', content)
                gpus = re.search(r'#SBATCH\s+--gres=gpu.*:(\d+)', content)
                time = re.search(r'#SBATCH\s+--time=(\d+):(\d+):(\d+)', content)
                
                print(f"\n{script.name}:")
                if cpus:
                    cpu_count = int(cpus.group(1))
                    if cpu_count <= 64:
                        print_check(f"  CPU: {cpu_count} 核", True)
                    else:
                        print_check(f"  CPU: {cpu_count} 核", False, "可能超过单节点限制")
                        self.warnings.append(f"{script.name}: CPU 请求可能过高")
                
                if mem:
                    mem_gb = int(mem.group(1))
                    if mem_gb <= 512:
                        print_check(f"  内存: {mem_gb}GB", True)
                    else:
                        print_check(f"  内存: {mem_gb}GB", False, "可能超过节点限制")
                        self.warnings.append(f"{script.name}: 内存请求可能过高")
                
                if gpus:
                    gpu_count = int(gpus.group(1))
                    print_check(f"  GPU: {gpu_count} 个", True)
                
                if time:
                    hours = int(time.group(1))
                    print_check(f"  时间限制: {hours} 小时", True)
                    
    def check_model_paths(self):
        """检查模型路径配置"""
        # 检查是否使用共享模型库
        env_vars = [
            "HF_HOME=/home/share/models",
            "TRANSFORMERS_CACHE=/home/share/models",
        ]
        
        slurm_files = list(self.project_root.glob("scripts/*.sh"))
        uses_shared = False
        
        for script in slurm_files:
            with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if any(var in content for var in env_vars):
                    uses_shared = True
                    print_check(f"{script.name}: 使用共享模型库", True)
                    break
        
        if not uses_shared:
            self.warnings.append("未配置共享模型库，首次运行可能需要下载模型")
            print_check("共享模型库配置", False, "建议配置 HF_HOME=/home/share/models")
            
    def check_data_paths(self):
        """检查数据路径"""
        data_dir = self.project_root / "data"
        if data_dir.exists():
            print_check("data/ 目录", True)
            
            # 检查数据切分
            splits_dir = data_dir / "unified_splits"
            if splits_dir.exists():
                print_check("  unified_splits/", True)
            else:
                print_check("  unified_splits/", False, "需要运行 data_split_controller.py")
                self.warnings.append("数据切分未创建")
        else:
            print_check("data/ 目录", False, "目录不存在")
            self.errors.append("data/ 目录缺失")
            
    def print_summary(self):
        """打印最终总结"""
        print_header("验证总结")
        
        if not self.errors and not self.warnings:
            print(f"{GREEN}✓ 所有检查通过！项目已准备好进行训练。{RESET}")
            print("\n下一步:")
            print("  1. 在登录节点运行:")
            print("     bash scripts/verify_login_node.sh")
            print("  2. 提交 GPU 检测:")
            print("     sbatch scripts/check_gpu_node.sh")
            print("  3. 查看 GPU 报告后提交训练:")
            print("     sbatch scripts/run_multi_seed_experiments.sh")
            return 0
        
        if self.errors:
            print(f"\n{RED}✗ 发现 {len(self.errors)} 个错误:{RESET}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}⚠ 发现 {len(self.warnings)} 个警告:{RESET}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if self.errors:
            print(f"\n{RED}请修复错误后再提交训练任务！{RESET}")
            return 1
        else:
            print(f"\n{YELLOW}警告不会阻止训练，但建议检查和修复。{RESET}")
            return 0


def main():
    validator = KaVaPreTrainingValidator()
    exit_code = validator.check_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
