"""
Analyze training results from KaVa experiments E1-E5.

Reads training logs and generates comparison report to validate three claims:
1. KV compression preserves supervision signal
2. KV alignment supplements latent supervision  
3. R-KV provides best stability

Usage:
    python scripts/analyze_results.py --output_base outputs/
"""
import os
import re
import argparse
import json
from pathlib import Path


def parse_log_file(log_path):
    """Parse training log and extract loss values per epoch."""
    losses = []
    if not os.path.exists(log_path):
        return losses
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match: Epoch X/Y done. Avg Loss=Z.ZZZZ, Last CE=A.AAAA, KV=B.BBBB, CODI=C.CCCC
            match = re.search(r'Epoch (\d+)/\d+ done\. Avg Loss=([\d.]+), Last CE=([\d.]+), KV=([\d.]+), CODI=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                avg_loss = float(match.group(2))
                ce = float(match.group(3))
                kv = float(match.group(4))
                codi = float(match.group(5))
                losses.append({
                    'epoch': epoch,
                    'avg_loss': avg_loss,
                    'ce': ce,
                    'kv': kv,
                    'codi': codi
                })
    return losses


def analyze_experiment(exp_dir):
    """Analyze single experiment directory."""
    log_path = os.path.join(exp_dir, 'training_log.txt')
    completion_path = os.path.join(exp_dir, 'TRAINING_COMPLETED.txt')
    
    result = {
        'name': os.path.basename(exp_dir),
        'completed': os.path.exists(completion_path),
        'losses': parse_log_file(log_path),
        'final_ce': None,
        'final_kv': None,
        'final_codi': None,
        'final_avg': None
    }
    
    if result['losses']:
        last = result['losses'][-1]
        result['final_ce'] = last['ce']
        result['final_kv'] = last['kv']
        result['final_codi'] = last['codi']
        result['final_avg'] = last['avg_loss']
    
    return result


def compare_experiments(results):
    """Generate comparison report."""
    report = []
    report.append("=" * 80)
    report.append("KaVa Experiment Results - Validation of Three Claims")
    report.append("=" * 80)
    report.append("")
    
    # Table header
    report.append(f"{'Experiment':<25} {'Status':<12} {'Avg Loss':<12} {'CE Loss':<12} {'KV Loss':<12} {'CODI':<12}")
    report.append("-" * 80)
    
    for r in results:
        status = "✓ Done" if r['completed'] else "✗ Failed"
        avg = f"{r['final_avg']:.4f}" if r['final_avg'] else "N/A"
        ce = f"{r['final_ce']:.4f}" if r['final_ce'] else "N/A"
        kv = f"{r['final_kv']:.4f}" if r['final_kv'] else "N/A"
        codi = f"{r['final_codi']:.4f}" if r['final_codi'] else "N/A"
        report.append(f"{r['name']:<25} {status:<12} {avg:<12} {ce:<12} {kv:<12} {codi:<12}")
    
    report.append("=" * 80)
    report.append("")
    
    # Analysis of three claims
    report.append("CLAIM VALIDATION:")
    report.append("")
    
    # Find experiments
    e1 = next((r for r in results if 'baseline' in r['name'].lower()), None)
    e2 = next((r for r in results if 'full_kv' in r['name'].lower()), None)
    e3 = next((r for r in results if 'right_crop' in r['name'].lower()), None)
    e4 = next((r for r in results if 'rkv' in r['name'].lower() and 'shuffled' not in r['name'].lower()), None)
    e5 = next((r for r in results if 'shuffled' in r['name'].lower()), None)
    
    # Claim 1: KV compression preserves supervision
    report.append("1. KV Compression Preserves Supervision Signal")
    if e1 and e2 and e1['final_avg'] and e2['final_avg']:
        improvement = ((e1['final_avg'] - e2['final_avg']) / e1['final_avg']) * 100
        report.append(f"   Baseline (E1): {e1['final_avg']:.4f}")
        report.append(f"   Full KV (E2):  {e2['final_avg']:.4f}")
        report.append(f"   → Improvement: {improvement:+.2f}%")
        report.append(f"   {'✓ VALIDATED' if improvement > 0 else '✗ NOT VALIDATED'}")
    else:
        report.append("   ⚠ Insufficient data")
    report.append("")
    
    # Claim 2: KV alignment supplements latent supervision
    report.append("2. KV Alignment Supplements Latent Supervision")
    if e1 and e4 and e1['final_avg'] and e4['final_avg']:
        improvement = ((e1['final_avg'] - e4['final_avg']) / e1['final_avg']) * 100
        report.append(f"   Baseline (E1): {e1['final_avg']:.4f}")
        report.append(f"   R-KV (E4):     {e4['final_avg']:.4f}")
        report.append(f"   → Improvement: {improvement:+.2f}%")
        report.append(f"   {'✓ VALIDATED' if improvement > 0 else '✗ NOT VALIDATED'}")
    else:
        report.append("   ⚠ Insufficient data")
    report.append("")
    
    # Claim 3: R-KV provides best stability
    report.append("3. R-KV Provides Best Stability")
    if e2 and e3 and e4 and all(x['final_avg'] for x in [e2, e3, e4]):
        report.append(f"   Full KV (E2):       {e2['final_avg']:.4f}")
        report.append(f"   Right-crop (E3):    {e3['final_avg']:.4f}")
        report.append(f"   R-KV (E4):          {e4['final_avg']:.4f}")
        best = min(e2['final_avg'], e3['final_avg'], e4['final_avg'])
        report.append(f"   → Best: {best:.4f}")
        if abs(e4['final_avg'] - best) < 0.001:
            report.append("   ✓ VALIDATED: R-KV achieves best or competitive loss")
        else:
            report.append("   ⚠ PARTIAL: R-KV not optimal but may have better variance")
    else:
        report.append("   ⚠ Insufficient data")
    report.append("")
    
    # Control experiment
    report.append("4. Negative Control (Shuffled KV)")
    if e4 and e5 and e4['final_avg'] and e5['final_avg']:
        degradation = ((e5['final_avg'] - e4['final_avg']) / e4['final_avg']) * 100
        report.append(f"   R-KV (E4):        {e4['final_avg']:.4f}")
        report.append(f"   Shuffled (E5):    {e5['final_avg']:.4f}")
        report.append(f"   → Degradation: {degradation:+.2f}%")
        report.append(f"   {'✓ As expected' if degradation > 0 else '✗ Unexpected result'}")
    else:
        report.append("   ⚠ Insufficient data")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base", type=str, default="outputs",
                        help="Base directory containing experiment outputs")
    parser.add_argument("--experiments", nargs="+", 
                        default=["E1_baseline", "E2_full_kv", "E3_right_crop", "E4_rkv", "E5_shuffled"],
                        help="List of experiment directory names")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    # Analyze each experiment
    results = []
    for exp_name in args.experiments:
        exp_dir = os.path.join(args.output_base, exp_name)
        if os.path.exists(exp_dir):
            result = analyze_experiment(exp_dir)
            results.append(result)
        else:
            print(f"Warning: {exp_dir} not found")
    
    # Generate report
    report = compare_experiments(results)
    print(report)
    
    # Save report
    report_path = os.path.join(args.output_base, "EXPERIMENT_REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    
    # Save JSON if requested
    if args.save_json:
        json_path = args.save_json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
