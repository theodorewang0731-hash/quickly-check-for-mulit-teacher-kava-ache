"""
训练完成后显示可视化报告信息
可以在训练脚本末尾调用
"""

import sys
from pathlib import Path
import socket


def print_visualization_info(output_dir: str, username: str = None, hostname: str = None):
    """
    打印可视化报告的下载信息
    
    Args:
        output_dir: 实验输出目录
        username: HPC 用户名（自动检测）
        hostname: HPC 主机名（自动检测）
    """
    output_path = Path(output_dir)
    viz_dir = output_path / "visualizations"
    summary_html = viz_dir / "experiment_summary.html"
    
    # 自动检测用户名和主机名
    if username is None:
        import os
        username = os.environ.get('USER', 'your_username')
    
    if hostname is None:
        try:
            hostname = socket.gethostname()
        except:
            hostname = 'hpc_address'
    
    # 检查文件是否存在
    if not summary_html.exists():
        print(f"⚠ Visualization not found at: {summary_html}")
        return
    
    # 获取文件大小
    file_size = summary_html.stat().st_size / 1024  # KB
    
    # 打印信息
    print("\n" + "="*80)
    print("🎉 Training Complete! Visualization Report Ready")
    print("="*80)
    print(f"\n📊 Main Report: {summary_html.absolute()}")
    print(f"   Size: {file_size:.1f} KB (self-contained, images embedded)")
    print(f"\n{'='*80}")
    print("📥 COPY AND RUN THIS COMMAND ON YOUR LOCAL MACHINE:")
    print("="*80)
    
    # 生成下载命令
    scp_command = f"scp {username}@{hostname}:{summary_html.absolute()} ~/Downloads/experiment_report.html"
    
    print(f"\n  {scp_command}\n")
    
    print("="*80)
    print("Then open the file:")
    print("="*80)
    print("\n  # macOS:")
    print("  open ~/Downloads/experiment_report.html")
    print("\n  # Windows (PowerShell):")
    print("  start ~/Downloads/experiment_report.html")
    print("\n  # Linux:")
    print("  xdg-open ~/Downloads/experiment_report.html")
    print("\n" + "="*80)
    
    # 列出其他可用的报告
    html_files = list(viz_dir.glob("*.html"))
    if len(html_files) > 1:
        print("\n📋 Other Available Reports:")
        for html_file in sorted(html_files):
            if html_file != summary_html:
                size = html_file.stat().st_size / 1024
                print(f"   - {html_file.name} ({size:.1f} KB)")
        print(f"\n   Download all: scp -r {username}@{hostname}:{viz_dir.absolute()} ~/Downloads/")
    
    print("\n" + "="*80)
    print("💡 Tips:")
    print("="*80)
    print("  • The HTML file works offline (no internet needed)")
    print("  • You can share it via email or cloud storage")
    print("  • All images are embedded (no separate files needed)")
    print("  • Compatible with all modern browsers")
    print("="*80 + "\n")


def create_download_script(output_dir: str):
    """
    创建一个自动下载脚本
    """
    output_path = Path(output_dir)
    viz_dir = output_path / "visualizations"
    summary_html = viz_dir / "experiment_summary.html"
    
    if not summary_html.exists():
        return None
    
    import os
    username = os.environ.get('USER', 'your_username')
    hostname = socket.gethostname()
    
    script_path = output_path / "download_and_open.sh"
    
    script_content = f"""#!/bin/bash
# Auto-download and open experiment report
# Run this on your LOCAL machine

REPORT_FILE="experiment_report_$(date +%Y%m%d_%H%M%S).html"

echo "Downloading experiment report..."
scp {username}@{hostname}:{summary_html.absolute()} ~/Downloads/$REPORT_FILE

if [ $? -eq 0 ]; then
    echo "✓ Downloaded to: ~/Downloads/$REPORT_FILE"
    echo "Opening in browser..."
    
    # Detect OS and open
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open ~/Downloads/$REPORT_FILE
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open ~/Downloads/$REPORT_FILE
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        start ~/Downloads/$REPORT_FILE
    else
        echo "Please open ~/Downloads/$REPORT_FILE manually"
    fi
else
    echo "✗ Download failed. Please check:"
    echo "  1. SSH connection is working"
    echo "  2. File path is correct"
    echo "  3. You have permission to access the file"
fi
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)  # Make executable
    
    print(f"\n✓ Auto-download script created: {script_path}")
    print(f"\n  To use (on your local machine):")
    print(f"  1. scp {username}@{hostname}:{script_path.absolute()} ~/")
    print(f"  2. bash ~/download_and_open.sh")
    
    return script_path


def main():
    """命令行使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display visualization report info")
    parser.add_argument("output_dir", help="Experiment output directory")
    parser.add_argument("--username", help="HPC username (auto-detected if not provided)")
    parser.add_argument("--hostname", help="HPC hostname (auto-detected if not provided)")
    parser.add_argument("--create-script", action="store_true", help="Create auto-download script")
    
    args = parser.parse_args()
    
    print_visualization_info(args.output_dir, args.username, args.hostname)
    
    if args.create_script:
        create_download_script(args.output_dir)


if __name__ == "__main__":
    main()
