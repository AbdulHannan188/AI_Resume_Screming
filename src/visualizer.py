import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PALETTE = 'tab20'
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _save(fig, name: str):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


class Visualizer:

    # ------------------------------------------------------------------ #
    # 1. Category distribution bar chart
    # ------------------------------------------------------------------ #
    def plot_category_distribution(self, df: pd.DataFrame):
        counts = df['Category'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Resume Dataset — Category Distribution', fontsize=16, fontweight='bold')

        # Bar chart
        colors = plt.cm.get_cmap(PALETTE)(np.linspace(0, 1, len(counts)))
        bars = axes[0].barh(counts.index, counts.values, color=colors)
        axes[0].set_xlabel('Number of Resumes')
        axes[0].set_title('Resumes per Category')
        for bar, val in zip(bars, counts.values):
            axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         str(val), va='center', fontsize=9)
        axes[0].invert_yaxis()

        # Pie chart (top 10 + Other)
        top10 = counts.head(10)
        other = counts.iloc[10:].sum()
        pie_vals = list(top10.values) + ([other] if other > 0 else [])
        pie_labels = list(top10.index) + (['Other'] if other > 0 else [])
        wedge_colors = plt.cm.get_cmap(PALETTE)(np.linspace(0, 1, len(pie_vals)))
        axes[1].pie(pie_vals, labels=pie_labels, colors=wedge_colors,
                    autopct='%1.1f%%', startangle=140,
                    textprops={'fontsize': 8}, pctdistance=0.82)
        axes[1].set_title('Category Share (top 10)')

        plt.tight_layout()
        _save(fig, '01_category_distribution.png')

    # ------------------------------------------------------------------ #
    # 2. Resume length distribution
    # ------------------------------------------------------------------ #
    def plot_resume_length_distribution(self, df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Resume Length Analysis', fontsize=15, fontweight='bold')

        char_lengths = df['Resume'].str.len()
        word_counts = df['Resume'].str.split().str.len()

        axes[0].hist(char_lengths, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        axes[0].axvline(char_lengths.mean(), color='crimson', linestyle='--', label=f'Mean: {char_lengths.mean():.0f}')
        axes[0].set_xlabel('Character Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Character Length Distribution')
        axes[0].legend()

        axes[1].hist(word_counts, bins=40, color='darkorange', edgecolor='white', alpha=0.85)
        axes[1].axvline(word_counts.mean(), color='crimson', linestyle='--', label=f'Mean: {word_counts.mean():.0f}')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Word Count Distribution')
        axes[1].legend()

        plt.tight_layout()
        _save(fig, '02_resume_length_distribution.png')

    # ------------------------------------------------------------------ #
    # 3. Top words per selected categories
    # ------------------------------------------------------------------ #
    def plot_word_frequency(self, df: pd.DataFrame, n_words: int = 15, n_cats: int = 6):
        stop = {'experience', 'work', 'years', 'year', 'skills', 'knowledge',
                'proficient', 'using', 'used', 'use', 'also', 'ability',
                'strong', 'good', 'excellent', 'various'}
        categories = df['Category'].value_counts().head(n_cats).index.tolist()

        cols = 3
        rows = (len(categories) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
        axes = np.array(axes).flatten()
        fig.suptitle(f'Top {n_words} Words by Category', fontsize=15, fontweight='bold')

        for i, cat in enumerate(categories):
            texts = ' '.join(df[df['Category'] == cat]['Resume'].str.lower())
            words = re.findall(r'\b[a-z]{3,}\b', texts)
            freq = Counter(w for w in words if w not in stop)
            top = freq.most_common(n_words)
            if not top:
                axes[i].axis('off')
                continue
            labels, vals = zip(*top)
            color = plt.cm.get_cmap(PALETTE)(i / n_cats)
            axes[i].barh(labels[::-1], vals[::-1], color=color)
            axes[i].set_title(cat, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Frequency')

        for j in range(len(categories), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        _save(fig, '03_word_frequency_by_category.png')

    # ------------------------------------------------------------------ #
    # 4. Model accuracy comparison
    # ------------------------------------------------------------------ #
    def plot_model_comparison(self, results: dict):
        names = list(results.keys())
        accs = [results[n]['accuracy'] for n in names]
        cv_means = [results[n]['cv_mean'] for n in names]
        cv_stds = [results[n]['cv_std'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle('ML Model Performance Comparison', fontsize=15, fontweight='bold')

        bars1 = ax.bar(x - width / 2, accs, width, label='Test Accuracy',
                       color='steelblue', alpha=0.85)
        bars2 = ax.bar(x + width / 2, cv_means, width, label='CV Mean Accuracy',
                       color='darkorange', alpha=0.85, yerr=cv_stds, capsize=5)

        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title('Test Accuracy vs Cross-Validation Accuracy')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

        best = names[np.argmax(accs)]
        ax.annotate(f'Best: {best}', xy=(np.argmax(accs) - width / 2, max(accs)),
                    xytext=(np.argmax(accs) - width / 2 + 0.5, max(accs) + 0.05),
                    arrowprops=dict(arrowstyle='->', color='crimson'),
                    color='crimson', fontsize=11)

        plt.tight_layout()
        _save(fig, '04_model_comparison.png')

    # ------------------------------------------------------------------ #
    # 5. Confusion matrix for best model
    # ------------------------------------------------------------------ #
    def plot_confusion_matrix(self, model, X_test, y_test, label_encoder):
        y_pred = model.predict(X_test)
        classes = label_encoder.classes_
        n = len(classes)

        from sklearn.metrics import confusion_matrix as cm_fn
        cm = cm_fn(y_test, y_pred)

        fig_size = max(10, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix — Best Model', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        _save(fig, '05_confusion_matrix.png')

    # ------------------------------------------------------------------ #
    # 6. Screening results — match scores
    # ------------------------------------------------------------------ #
    def plot_screening_results(self, results: pd.DataFrame, job_title: str, save_suffix: str = ''):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Resume Screening Results\n"{job_title}"',
                     fontsize=13, fontweight='bold')

        # Horizontal bar — match %
        labels = [f"#{r['Rank']} {r['Category']}" for _, r in results.iterrows()]
        scores = results['Match_%'].values
        colors_map = plt.cm.get_cmap('RdYlGn')(scores / 100)
        axes[0].barh(labels[::-1], scores[::-1], color=colors_map[::-1])
        axes[0].set_xlabel('Match Score (%)')
        axes[0].set_title('Top Matching Resumes by Score')
        for i, (lbl, sc) in enumerate(zip(labels[::-1], scores[::-1])):
            axes[0].text(sc + 0.2, i, f'{sc:.1f}%', va='center', fontsize=9)
        axes[0].set_xlim(0, max(scores) * 1.15 + 1)

        # Category breakdown of top matches
        cat_counts = results['Category'].value_counts()
        cat_colors = plt.cm.get_cmap(PALETTE)(np.linspace(0, 1, len(cat_counts)))
        axes[1].pie(cat_counts.values, labels=cat_counts.index, colors=cat_colors,
                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
        predicted = results['Predicted_JD_Category'].iloc[0]
        axes[1].set_title(f'Category Mix\n(JD predicted as: {predicted})')

        plt.tight_layout()
        safe = re.sub(r'[^\w]', '_', job_title)[:30]
        _save(fig, f'06_screening_{safe}{save_suffix}.png')

    # ------------------------------------------------------------------ #
    # 7. Word cloud for a category
    # ------------------------------------------------------------------ #
    def plot_wordcloud(self, df: pd.DataFrame, category: str = None):
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("  wordcloud not installed — skipping word cloud chart.")
            return

        if category:
            subset = df[df['Category'] == category]['Resume']
            title = f'Word Cloud — {category}'
            fname = f'07_wordcloud_{re.sub(chr(32), "_", category)}.png'
        else:
            subset = df['Resume']
            title = 'Word Cloud — All Resumes'
            fname = '07_wordcloud_all.png'

        text = ' '.join(subset.str.lower())
        wc = WordCloud(width=900, height=500, background_color='white',
                       colormap='viridis', max_words=150,
                       collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        _save(fig, fname)

    # ------------------------------------------------------------------ #
    # 8. Per-category F1 score bar chart
    # ------------------------------------------------------------------ #
    def plot_f1_by_category(self, results: dict, best_model_name: str):
        from sklearn.metrics import f1_score
        model_res = results[best_model_name]
        report_lines = model_res['report'].strip().split('\n')

        categories, f1_scores = [], []
        for line in report_lines[2:]:
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ('accuracy', 'macro', 'weighted'):
                cat = ' '.join(parts[:-4])
                try:
                    f1_scores.append(float(parts[-3]))
                    categories.append(cat if cat else parts[0])
                except ValueError:
                    pass

        if not categories:
            return

        order = np.argsort(f1_scores)
        cats_sorted = [categories[i] for i in order]
        f1_sorted = [f1_scores[i] for i in order]

        fig, ax = plt.subplots(figsize=(11, max(6, len(cats_sorted) * 0.4)))
        colors = plt.cm.get_cmap('RdYlGn')(np.array(f1_sorted))
        ax.barh(cats_sorted, f1_sorted, color=colors)
        ax.axvline(0.8, color='crimson', linestyle='--', alpha=0.6, label='0.80 threshold')
        ax.set_xlabel('F1-Score')
        ax.set_title(f'Per-Category F1-Score — {best_model_name}', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.legend()
        for i, (cat, val) in enumerate(zip(cats_sorted, f1_sorted)):
            ax.text(val + 0.005, i, f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        _save(fig, '08_f1_by_category.png')
