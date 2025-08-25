import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Your actual data from the evaluation
data = {
    "HuBERT Hungarian": {
        "mrr": 0.48,
        "recall_at_1": 0.38,
        "recall_at_3": 0.6,
        "qps": 18.31,
        "avg_time": 0.0544,
        "total_time": 5.89,
        "build_time": 3.15,
        "search_time": 2.73
    },
    "BGE-M3": {
        "mrr": 0.903,
        "recall_at_1": 0.86,
        "recall_at_3": 0.96,
        "qps": 6.11,
        "avg_time": 0.1635,
        "total_time": 22.53,
        "build_time": 14.35,
        "search_time": 8.18
    },
    "SentenceTransformer Multilingual": {
        "mrr": 0.84,
        "recall_at_1": 0.78,
        "recall_at_3": 0.92,
        "qps": 40.04,
        "avg_time": 0.0248,
        "total_time": 2.41,
        "build_time": 1.16,
        "search_time": 1.25
    },
    "OpenAI_Ada-002": {
        "mrr": 0.8,
        "recall_at_1": 0.72,
        "recall_at_3": 0.9,
        "qps": 3.20,
        "avg_time": 0.3123,
        "total_time": 16.54,
        "build_time": 0.92,
        "search_time": 15.62
    },
    "OpenAI_3-Small": {
        "mrr": 0.807,
        "recall_at_1": 0.7,
        "recall_at_3": 0.94,
        "qps": 2.97,
        "avg_time": 0.3367,
        "total_time": 17.71,
        "build_time": 0.87,
        "search_time": 16.84
    },
    "Ollama_Nomic": {
        "mrr": 0.71,
        "recall_at_1": 0.64,
        "recall_at_3": 0.8,
        "qps": 16.20,
        "avg_time": 0.0617,
        "total_time": 10.52,
        "build_time": 7.42,
        "search_time": 3.09
    },
    "Ollama_MiniLM": {
        "mrr": 0.59,
        "recall_at_1": 0.46,
        "recall_at_3": 0.74,
        "qps": 33.39,
        "avg_time": 0.0299,
        "total_time": 3.75,
        "build_time": 2.25,
        "search_time": 1.50
    }
}


def print_summary_rankings():
    """Print comprehensive rankings"""
    print("🏆 HUNGARIAN EMBEDDING MODELS - EVALUATION RESULTS")
    print("=" * 80)
    print(f"📅 Evaluation Date: 2025-08-24 16:35:24")
    print(f"⏱️ Total Evaluation Time: 92.8 seconds")
    print(f"🔢 Models Tested: {len(data)}")
    print(f"📝 Test Questions: 50")
    print()

    # Performance Rankings (MRR)
    print("🎯 PERFORMANCE RANKINGS (by MRR):")
    print("-" * 80)
    sorted_by_mrr = sorted(data.items(), key=lambda x: x[1]['mrr'], reverse=True)
    for i, (model, metrics) in enumerate(sorted_by_mrr, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(
            f"{emoji} {model:<35} MRR: {metrics['mrr']:.3f} | R@1: {metrics['recall_at_1']:.3f} | R@3: {metrics['recall_at_3']:.3f}")

    print(f"\n⚡ SPEED RANKINGS (by QPS):")
    print("-" * 80)
    sorted_by_speed = sorted(data.items(), key=lambda x: x[1]['qps'], reverse=True)
    for i, (model, metrics) in enumerate(sorted_by_speed, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"{emoji} {model:<35} {metrics['qps']:6.1f} QPS | {metrics['avg_time']:.4f}s per query")

    # Efficiency Rankings
    efficiency_scores = {model: metrics['mrr'] / metrics['avg_time'] for model, metrics in data.items()}
    sorted_by_efficiency = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🚀 EFFICIENCY RANKINGS (MRR/Time):")
    print("-" * 80)
    for i, (model, efficiency) in enumerate(sorted_by_efficiency, 1):
        emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        mrr = data[model]['mrr']
        time_per_query = data[model]['avg_time']
        print(f"{emoji} {model:<35} {efficiency:6.1f} (MRR: {mrr:.3f}, Time: {time_per_query:.4f}s)")

    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 80)
    best_mrr = max(data.items(), key=lambda x: x[1]['mrr'])
    fastest = max(data.items(), key=lambda x: x[1]['qps'])
    most_efficient = max(efficiency_scores.items(), key=lambda x: x[1])

    print(f"🏆 Best Accuracy: {best_mrr[0]} (MRR: {best_mrr[1]['mrr']:.3f})")
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['qps']:.1f} QPS)")
    print(f"🎯 Most Efficient: {most_efficient[0]} (Score: {most_efficient[1]:.1f})")

    slowest_qps = min(data.items(), key=lambda x: x[1]['qps'])
    speed_ratio = fastest[1]['qps'] / slowest_qps[1]['qps']
    print(f"📊 Speed Difference: {speed_ratio:.1f}x ({fastest[0]} vs {slowest_qps[0]})")


def plot_1_performance_vs_speed():
    """Performance vs Speed Scatter Plot"""
    models = list(data.keys())
    short_names = [name.split()[0] if len(name.split()) > 1 else name for name in models]
    mrr_scores = [data[model]['mrr'] for model in models]
    qps_scores = [data[model]['qps'] for model in models]

    plt.figure(figsize=(14, 10))

    # Create scatter plot with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    scatter = plt.scatter(qps_scores, mrr_scores, s=400, alpha=0.8, c=colors,
                          edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, (model, qps, mrr) in enumerate(zip(short_names, qps_scores, mrr_scores)):
        plt.annotate(model, (qps, mrr), xytext=(8, 8), textcoords='offset points',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[i], alpha=0.7, edgecolor='black'))

    # Add quadrant lines
    mean_qps = np.mean(qps_scores)
    mean_mrr = np.mean(mrr_scores)
    plt.axhline(y=mean_mrr, color='red', linestyle='--', alpha=0.6, linewidth=2)
    plt.axvline(x=mean_qps, color='red', linestyle='--', alpha=0.6, linewidth=2)

    # Quadrant labels
    plt.text(0.02, 0.98, 'High Accuracy\nLow Speed', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=12, fontweight='bold')
    plt.text(0.98, 0.98, '🏆 OPTIMAL ZONE\nHigh Accuracy\nHigh Speed', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=12, fontweight='bold')
    plt.text(0.02, 0.02, 'Low Accuracy\nLow Speed', transform=plt.gca().transAxes,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontsize=12, fontweight='bold')
    plt.text(0.98, 0.02, 'Low Accuracy\nHigh Speed', transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=12, fontweight='bold')

    plt.xlabel('Queries Per Second (Higher is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('MRR Score (Higher is Better)', fontsize=14, fontweight='bold')
    plt.title('Hungarian Embedding Models: Performance vs Speed\n(Top-right quadrant is optimal)',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # Add some styling
    plt.tight_layout()
    plt.show()


def plot_2_comprehensive_metrics():
    """All Metrics Comparison Bar Chart"""
    models = list(data.keys())
    short_names = [name.replace('SentenceTransformer', 'ST').replace('Hungarian', 'HU') for name in models]

    mrr_scores = [data[model]['mrr'] for model in models]
    recall_1_scores = [data[model]['recall_at_1'] for model in models]
    recall_3_scores = [data[model]['recall_at_3'] for model in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 10))

    bars1 = ax.bar(x - width, mrr_scores, width, label='MRR',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, recall_1_scores, width, label='Recall@1',
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, recall_3_scores, width, label='Recall@3',
                   color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Hungarian Embedding Models - All Performance Metrics Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


def plot_3_timing_breakdown():
    """Timing Breakdown Stacked Bar Chart"""
    models = list(data.keys())
    short_names = [name.split()[0] if len(name.split()) > 1 else name[:20] for name in models]

    build_times = [data[model]['build_time'] for model in models]
    search_times = [data[model]['search_time'] for model in models]

    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(models))
    width = 0.6

    bars1 = ax.bar(x, build_times, width, label='Build Time (Embedding + Setup)',
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, search_times, width, bottom=build_times,
                   label='Search Phase Time', color='#E74C3C', alpha=0.8,
                   edgecolor='black', linewidth=1)

    # Add total time labels on top
    for i, model in enumerate(models):
        total_time = data[model]['total_time']
        ax.text(i, total_time + max(build_times + search_times) * 0.02,
                f'{total_time:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Timing Breakdown by Model Components\n(Total time shown on top)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_4_efficiency_ranking():
    """Efficiency Score Horizontal Bar Chart"""
    models = list(data.keys())
    efficiency_scores = [data[model]['mrr'] / data[model]['avg_time'] for model in models]

    # Sort by efficiency
    sorted_data = sorted(zip(models, efficiency_scores), key=lambda x: x[1], reverse=True)
    sorted_models, sorted_scores = zip(*sorted_data)

    short_names = [name.replace('SentenceTransformer', 'ST').replace('Hungarian', 'HU')
                   for name in sorted_models]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_models)))
    bars = ax.barh(range(len(sorted_models)), sorted_scores, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1)

    # Add score labels and rankings
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score + max(sorted_scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.1f}', va='center', fontweight='bold', fontsize=11)

        # Add ranking medal/number
        if i == 0:
            medal = "🥇"
        elif i == 1:
            medal = "🥈"
        elif i == 2:
            medal = "🥉"
        else:
            medal = f"#{i + 1}"

        ax.text(max(sorted_scores) * 0.02, bar.get_y() + bar.get_height() / 2,
                medal, va='center', ha='left', fontsize=14, fontweight='bold')

    ax.set_xlabel('Efficiency Score (MRR / Average Query Time)', fontsize=14, fontweight='bold')
    ax.set_title('Model Efficiency Ranking\n(Higher Score = Better Performance per Unit Time)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(short_names)
    ax.grid(axis='x', alpha=0.3)

    plt.gca().invert_yaxis()  # Highest efficiency on top
    plt.tight_layout()
    plt.show()


def plot_5_query_speed_comparison():
    """Query Processing Speed Comparison"""
    models = list(data.keys())
    avg_times = [data[model]['avg_time'] for model in models]
    qps_scores = [data[model]['qps'] for model in models]

    # Sort by average time (ascending - fastest first)
    sorted_data = sorted(zip(models, avg_times, qps_scores), key=lambda x: x[1])
    sorted_models, sorted_times, sorted_qps = zip(*sorted_data)

    short_names = [name.replace('SentenceTransformer', 'ST').replace('Hungarian', 'HU')
                   for name in sorted_models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left plot: Average time per query
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
    bars1 = ax1.barh(range(len(sorted_models)), sorted_times, color=colors,
                     alpha=0.8, edgecolor='black', linewidth=1)

    for i, (bar, time_val, qps_val) in enumerate(zip(bars1, sorted_times, sorted_qps)):
        ax1.text(time_val + max(sorted_times) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{time_val:.3f}s', va='center', fontweight='bold')

        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i + 1}"
        ax1.text(max(sorted_times) * 0.02, bar.get_y() + bar.get_height() / 2,
                 rank_emoji, va='center', ha='left', fontweight='bold', fontsize=12)

    ax1.set_xlabel('Average Time per Query (seconds)\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Query Processing Speed Ranking', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(sorted_models)))
    ax1.set_yticklabels(short_names)
    ax1.grid(axis='x', alpha=0.3)

    # Right plot: Queries per second (reverse order for consistency)
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_models)))
    bars2 = ax2.barh(range(len(sorted_models)), sorted_qps[::-1], color=colors2,
                     alpha=0.8, edgecolor='black', linewidth=1)

    for i, (bar, qps_val) in enumerate(zip(bars2, sorted_qps[::-1])):
        ax2.text(qps_val + max(sorted_qps) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{qps_val:.1f}', va='center', fontweight='bold')

    ax2.set_xlabel('Queries Per Second\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Ranking', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(sorted_models)))
    ax2.set_yticklabels(short_names[::-1])  # Reverse order
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_6_comprehensive_heatmap():
    """Comprehensive Heatmap of All Metrics"""
    models = list(data.keys())
    short_names = [name.replace('SentenceTransformer', 'ST').replace('Hungarian', 'HU')[:15]
                   for name in models]

    metrics = ['mrr', 'recall_at_1', 'recall_at_3', 'qps', 'avg_time']
    metric_labels = ['MRR', 'Recall@1', 'Recall@3', 'QPS', 'Avg Time\n(inverted)']

    # Create matrix - normalize data for heatmap
    data_matrix = []
    for model in models:
        row = []
        for metric in metrics:
            value = data[model][metric]
            if metric == 'avg_time':
                # For time metrics, lower is better - invert and normalize
                max_time = max(data[m]['avg_time'] for m in models)
                min_time = min(data[m]['avg_time'] for m in models)
                normalized = 1 - ((value - min_time) / (max_time - min_time)) if max_time > min_time else 1.0
            else:
                # For other metrics, higher is better
                max_val = max(data[m][metric] for m in models)
                min_val = min(data[m][metric] for m in models)
                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 1.0
            row.append(normalized)
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set labels
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=0, ha='center', fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short_names)

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Normalized Performance\n(Green = Better)', rotation=270, labelpad=20, fontweight='bold')

    # Add actual values as text
    for i in range(len(models)):
        for j, metric in enumerate(metrics):
            actual_value = data[models[i]][metric]
            if metric == 'qps':
                text = f'{actual_value:.1f}'
            elif metric == 'avg_time':
                text = f'{actual_value:.3f}s'
            else:
                text = f'{actual_value:.3f}'

            # Choose text color based on background
            text_color = 'black' if data_matrix[i][j] > 0.5 else 'white'
            ax.text(j, i, text, ha="center", va="center",
                    color=text_color, fontweight='bold', fontsize=9)

    ax.set_title('Comprehensive Performance Heatmap\n(Green = Better Performance)',
                 fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    plt.show()


def create_comprehensive_report():
    """Create all visualizations and analysis"""
    print_summary_rankings()
    print("\n" + "=" * 80)
    print("🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    print("=" * 80)

    print("\n1. 🎯 Performance vs Speed Analysis...")
    plot_1_performance_vs_speed()

    print("2. 📊 Comprehensive Metrics Comparison...")
    plot_2_comprehensive_metrics()

    print("3. ⏱️ Timing Breakdown Analysis...")
    plot_3_timing_breakdown()

    print("4. 🚀 Efficiency Ranking...")
    plot_4_efficiency_ranking()

    print("5. ⚡ Speed Comparison Analysis...")
    plot_5_query_speed_comparison()

    print("6. 🌈 Comprehensive Performance Heatmap...")
    plot_6_comprehensive_heatmap()

    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("=" * 80)

    # Final recommendations
    print("\n🎯 FINAL RECOMMENDATIONS:")
    print("-" * 80)
    print("🏆 For MAXIMUM ACCURACY: BGE-M3 (0.903 MRR)")
    print("⚡ For MAXIMUM SPEED: SentenceTransformer Multilingual (40.0 QPS)")
    print("🎯 For BEST BALANCE: SentenceTransformer Multilingual (33.9 efficiency)")
    print("💰 For LOCAL DEPLOYMENT: Ollama MiniLM (33.4 QPS, good efficiency)")
    print("🇭🇺 For HUNGARIAN FOCUS: Consider improving HuBERT or using multilingual models")
    print("\n💡 The SentenceTransformer Multilingual model offers the best")
    print("   overall performance/speed trade-off for Hungarian text processing!")


if __name__ == "__main__":
    # Run the comprehensive analysis
    #create_comprehensive_report()

    plot_3_timing_breakdown()