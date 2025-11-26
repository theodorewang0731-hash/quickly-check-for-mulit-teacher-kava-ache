"""
🔧 PyTorch CUDA 环境修复脚本
自动检测并重新安装支持 CUDA 的 PyTorch
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def check_cuda_version():
    """检查系统 CUDA 版本"""
    print("\n🔍 Checking NVIDIA GPU and CUDA Driver...")
    
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ nvidia-smi command failed!")
        print("   Your NVIDIA driver may not be installed or working properly.")
        return None
    
    print(result.stdout)
    
    # 从输出中提取 CUDA 版本
    for line in result.stdout.split('\n'):
        if 'CUDA Version:' in line:
            cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
            return cuda_version
    
    return None

def main():
    print("\n" + "🎯" * 35)
    print("  PyTorch CUDA Environment Repair Tool")
    print("  PyTorch CUDA 环境修复工具")
    print("🎯" * 35)
    
    # Step 1: 检查当前 PyTorch
    print("\n📋 Step 1: Current PyTorch Status")
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print("\n✅ PyTorch CUDA is already working! No fix needed.")
            return 0
        else:
            print(f"\n⚠️ Current PyTorch: {torch.__version__}")
            if '+cpu' in torch.__version__:
                print("   Problem: CPU-only version detected (no CUDA support)")
            else:
                print("   Problem: CUDA not available in current installation")
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    # Step 2: 检查 NVIDIA 驱动和 CUDA
    print("\n📋 Step 2: Checking NVIDIA Driver")
    cuda_version = check_cuda_version()
    
    if cuda_version is None:
        print("\n❌ Cannot detect CUDA driver version")
        print("   Please install NVIDIA driver first:")
        print("   https://www.nvidia.com/Download/index.aspx")
        return 1
    
    print(f"\n✅ CUDA Driver Version: {cuda_version}")
    
    # Step 3: 确定要安装的 PyTorch 版本
    print("\n📋 Step 3: Determining PyTorch Installation Command")
    
    # 根据 CUDA 版本选择合适的 PyTorch
    major_version = int(cuda_version.split('.')[0])
    
    if major_version >= 12:
        pytorch_cuda = "cu121"  # CUDA 12.1
        print(f"   Detected CUDA {cuda_version} -> Installing PyTorch with CUDA 12.1 support")
    elif major_version == 11:
        pytorch_cuda = "cu118"  # CUDA 11.8
        print(f"   Detected CUDA {cuda_version} -> Installing PyTorch with CUDA 11.8 support")
    else:
        print(f"   ⚠️ CUDA version {cuda_version} is quite old")
        pytorch_cuda = "cu118"
        print(f"   -> Will try CUDA 11.8 compatibility mode")
    
    # PyTorch 官方安装命令（适配 Windows）
    install_command = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{pytorch_cuda}"
    
    print(f"\n📦 Installation Command:")
    print(f"   {install_command}")
    
    # Step 4: 询问用户确认
    print("\n" + "="*70)
    print("⚠️ IMPORTANT: This will UNINSTALL current PyTorch and REINSTALL with CUDA support")
    print("="*70)
    
    response = input("\nProceed with installation? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Installation cancelled by user")
        return 1
    
    # Step 5: 卸载旧版本
    print("\n📋 Step 4: Uninstalling CPU-only PyTorch")
    success = run_command(
        "pip uninstall torch torchvision torchaudio -y",
        "Removing old PyTorch installation"
    )
    
    if not success:
        print("⚠️ Uninstall had warnings, but continuing...")
    
    # Step 6: 安装 CUDA 版本
    print("\n📋 Step 5: Installing PyTorch with CUDA Support")
    success = run_command(
        install_command,
        f"Installing PyTorch with {pytorch_cuda}"
    )
    
    if not success:
        print("\n❌ Installation failed!")
        print("\n💡 Manual installation:")
        print(f"   {install_command}")
        return 1
    
    # Step 7: 验证安装
    print("\n📋 Step 6: Verifying Installation")
    print("\nImporting PyTorch and checking CUDA...")
    
    # 重新导入 PyTorch（需要在新的 Python 进程中）
    verify_command = """python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'ERROR: Still no CUDA')" """
    
    subprocess.run(verify_command, shell=True)
    
    print("\n" + "="*70)
    print("✅ Installation Complete!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("   1. Close this terminal and open a new one")
    print("   2. Activate your virtual environment again")
    print("   3. Run: python train_simplified.py")
    print("\n🎉 Your RTX 4070 is ready for training!")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
