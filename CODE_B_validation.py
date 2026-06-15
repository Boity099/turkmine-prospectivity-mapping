"""
CODE B: VALIDATION AND ACCURACY ASSESSMENT
Purpose: Perform accuracy assessment, confusion matrix, and field validation
Input: Field assay data, prospectivity scores
Output: Validation metrics, confusion matrices, bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import os

print("=" * 70)
print("CODE B: VALIDATION AND ACCURACY ASSESSMENT")
print("=" * 70)

# Field validation data
data = {
    'Sample_ID': ['TK_25_RC_371', 'TK_25_RC_357', 'TK_25_RC_376A', 'TK_25_RC_107',
                  'TK_25_RC_195', 'TK_25_RC_432', 'TK_25_RC_209', 'TK_25_RC_219',
                  'TK_25_RC_181', 'TK_25_RC_194', 'TK_25_RC_200', 'TK_25_RC_400',
                  'TK_25_RC_603', 'TK_25_RC_199', 'TK_25_RC_213', 'TK_25_RC_214'],
    'Au_gpt': [0.77, 2.40, 1.68, 0.29, 0.83, 0.51, 0.13, 0.70,
               0.00, 0.40, 0.02, 0.18, 1.15, 0.24, 0.39, 0.23],
    'Prospectivity_Score': [0.52, 0.68, 0.55, 0.44, 0.48, 0.47, 0.41, 0.53,
                            0.38, 0.45, 0.35, 0.42, 0.58, 0.43, 0.46, 0.44]
}

df = pd.DataFrame(data)
print(f"Loaded {len(df)} validation points")
print(f"Mineralised (Au > 0.5): {sum(df['Au_gpt'] > 0.5)}")
print(f"Non-mineralised: {sum(df['Au_gpt'] <= 0.5)}")

# Classify
hpz_threshold = 0.47
df['Predicted'] = df['Prospectivity_Score'].apply(lambda x: 'HPZ' if x > hpz_threshold else 'MPZ/LPZ')
df['Actual'] = df['Au_gpt'].apply(lambda x: 'HPZ' if x > 0.5 else 'MPZ/LPZ')

# Binary classification
y_true = [1 if au > 0.5 else 0 for au in df['Au_gpt']]
y_pred = [1 if score > hpz_threshold else 0 for score in df['Prospectivity_Score']]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

p_e = ((tp + fn)*(tp + fp) + (tn + fp)*(tn + fn)) / ((tp+tn+fp+fn)**2)
kappa = (accuracy - p_e) / (1 - p_e)

print(f"\nConfusion Matrix:")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"\nAccuracy: {accuracy:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"F1-Score: {f1:.3f}")
print(f"Kappa: {kappa:.3f}")

# Zone statistics
zone_stats = {}
for zone in ['HPZ', 'MPZ/LPZ']:
    mask = df['Actual'] == zone
    zone_stats[zone] = {'count': sum(mask), 'mean': df[mask]['Au_gpt'].mean() if sum(mask) > 0 else 0}

print(f"\nHPZ: {zone_stats['HPZ']['count']} samples, mean {zone_stats['HPZ']['mean']:.2f} g/t")
print(f"MPZ/LPZ: {zone_stats['MPZ/LPZ']['count']} samples, mean {zone_stats['MPZ/LPZ']['mean']:.2f} g/t")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 6))
zones = ['HPZ', 'MPZ/LPZ']
means = [zone_stats['HPZ']['mean'], zone_stats['MPZ/LPZ']['mean']]
colors = ['#d73027', '#1a9850']
bars = ax.bar(zones, means, color=colors, edgecolor='black', linewidth=1.5)
for bar, mean_val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{mean_val:.2f} g/t', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Mean Gold Grade (g/t)')
ax.set_title('Field Validation Results')
ax.set_ylim(0, max(means) + 0.3)
plt.tight_layout()
plt.savefig("Validation_Bar_Chart.png", dpi=300)
print("\nBar chart saved: Validation_Bar_Chart.png")
plt.show()

print("\nCODE B COMPLETED")
