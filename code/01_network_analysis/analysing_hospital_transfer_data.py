""" Descriptive Statistics - Questions answered: Which hospital sends the most transfers? 
Which hospital receives the most? 
Which hospital has the most diverse partnerships?"""

hospital_stats = df.groupby('transfer_provider').agg(
    total_sent=('transfer_type', 'count'),
    total_received=('received_provider', 'count'),
    unique_partners=('received_provider', 'nunique'),
    ed_ed=('transfer_type', lambda x: (x=='ED_ED').sum()),
    ed_adm=('transfer_type', lambda x: (x=='ED_ADM').sum()),
    adm_ed=('transfer_type', lambda x: (x=='ADM_ED').sum()),
    adm_adm=('transfer_type', lambda x: (x=='ADM_ADM').sum()),
    op_op=('transfer_type', lambda x: (x=='OP_OP').sum())
).reset_index()

"""visuals: Grouped Bar Chart: Sent vs Received per Hospital"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Calculate sent and received per hospital
sent = df.groupby('transfer_provider').size().reset_index(name='sent')
received = df.groupby('received_provider').size().reset_index(name='received')
balance = sent.merge(received, left_on='transfer_provider', right_on='received_provider')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(balance['transfer_provider']))
width = 0.35

bars1 = ax.bar(x - width/2, balance['sent'], width, label='Sent', color='#E63946')
bars2 = ax.bar(x + width/2, balance['received'], width, label='Received', color='#2A9D8F')

ax.set_title('Transfers Sent vs Received per Hospital', fontsize=16, fontweight='bold')
ax.set_xlabel('Hospital', fontsize=12)
ax.set_ylabel('Number of Transfers', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(balance['transfer_provider'], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/sent_vs_received.png', dpi=300)
plt.show()

"""Visual: Stacked Bar: Transfer Types per Hospital"""

# Count transfer types per hospital
stacked_data = df.groupby(['transfer_provider', 'transfer_type']).size().unstack(fill_value=0)

# Plot
colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E76F51']
stacked_data.plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)

plt.title('Transfer Types per Hospital (Sending)', fontsize=16, fontweight='bold')
plt.xlabel('Hospital', fontsize=12)
plt.ylabel('Number of Transfers', fontsize=12)
plt.legend(title='Transfer Type', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/transfer_types_stacked.png', dpi=300)
plt.show()

"""100% Stacked Bar: Proportion of Transfer Types"""
# Normalize to percentages
stacked_pct = stacked_data.div(stacked_data.sum(axis=1), axis=0) * 100

stacked_pct.plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)

plt.title('Transfer Type Proportions per Hospital (%)', fontsize=16, fontweight='bold')
plt.xlabel('Hospital', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.legend(title='Transfer Type', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.axhline(y=50, color='white', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('outputs/transfer_types_proportions.png', dpi=300)
plt.show()


"""Transfer Balance Analysis
Are hospitals net senders or net receivers?
Questions answered:

Is RQM01 predominantly sending or receiving?
Which hospitals act as specialist receiving centres?
Which hospitals are redistribution hubs?"""

# Sent vs Received per hospital
sent = df.groupby('transfer_provider').size().reset_index(name='sent')
received = df.groupby('received_provider').size().reset_index(name='received')

# Merge
balance = sent.merge(received, 
                     left_on='transfer_provider', 
                     right_on='received_provider')

# Calculate balance
balance['net_balance'] = balance['sent'] - balance['received']
balance['ratio'] = balance['sent'] / balance['received']

# Classify
balance['role'] = balance['net_balance'].apply(
    lambda x: 'Net Sender' if x > 0 else ('Net Receiver' if x < 0 else 'Balanced')
)

"""Visual: Net Transfer Balance"""

balance['net_balance'] = balance['sent'] - balance['received']
balance_sorted = balance.sort_values('net_balance')

colors = ['#E63946' if x > 0 else '#2A9D8F' for x in balance_sorted['net_balance']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(balance_sorted['transfer_provider'], balance_sorted['net_balance'], color=colors)

ax.axvline(x=0, color='black', linewidth=1.5)
ax.set_title('Hospital Transfer Balance\n(Positive = Net Sender | Negative = Net Receiver)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Net Balance (Sent - Received)', fontsize=12)
ax.set_ylabel('Hospital', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, balance_sorted['net_balance']):
    ax.text(val + (50 if val >= 0 else -50), 
            bar.get_y() + bar.get_height()/2,
            str(int(val)), va='center', 
            ha='left' if val >= 0 else 'right', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E63946', label='Net Sender'),
                   Patch(facecolor='#2A9D8F', label='Net Receiver')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('outputs/net_balance.png', dpi=300)
plt.show()

"""Chi-Square Test: Transfer Type Distribution
Is the distribution of transfer types different between hospitals?
Questions answered:

Does RQM01 have a significantly different transfer type mix vs RYJ01?
Are some hospitals more ED-focused vs outpatient-focused in transfers?"""

from scipy.stats import chi2_contingency

# Create contingency table
# Rows = hospitals, Columns = transfer types
contingency_table = pd.crosstab(
    df['transfer_provider'], 
    df['transfer_type']
)

# Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")
print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

"""Kruskal-Wallis Test: Transfer Volumes Across Hospitals
Are monthly transfer volumes significantly different between hospitals?
Why Kruskal-Wallis instead of ANOVA?

Transfer counts are likely not normally distributed
Non-parametric = no assumption about distribution
More appropriate for count data"""

from scipy.stats import kruskal

# Monthly transfer counts per hospital
monthly_by_hospital = df.groupby(
    ['transfer_provider', df['transfer_date'].dt.to_period('M')]
).size().reset_index(name='monthly_count')

# Separate groups for each hospital
groups = [
    monthly_by_hospital[monthly_by_hospital['transfer_provider'] == h]['monthly_count'].values
    for h in monthly_by_hospital['transfer_provider'].unique()
]

# Kruskal-Wallis test (non-parametric ANOVA)
stat, p_value = kruskal(*groups)

print(f"Kruskal-Wallis H: {stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

"""Post-Hoc Analysis: Which Hospitals Differ?
After Kruskal-Wallis, identify which specific hospital pairs differ:"""

from scipy.stats import mannwhitneyu
from itertools import combinations

hospitals = df['transfer_provider'].unique()
posthoc_results = []

for h1, h2 in combinations(hospitals, 2):
    group1 = monthly_by_hospital[monthly_by_hospital['transfer_provider'] == h1]['monthly_count']
    group2 = monthly_by_hospital[monthly_by_hospital['transfer_provider'] == h2]['monthly_count']
    
    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    posthoc_results.append({
        'hospital_1': h1,
        'hospital_2': h2,
        'statistic': round(stat, 4),
        'p_value': round(p_value, 6),
        'significant': 'Yes' if p_value < 0.05 else 'No'
    })

posthoc_df = pd.DataFrame(posthoc_results)
print("\nPairwise Comparisons (Mann-Whitney U):")
print(posthoc_df[posthoc_df['significant'] == 'Yes'])

"""Trend Analysis Per Hospital
Is each hospital's transfer volume increasing, decreasing or stable?"""

from scipy.stats import linregress

hospital_trends = []

for hospital in df['transfer_provider'].unique():
    # Yearly counts
    yearly = df[df['transfer_provider'] == hospital].groupby('financial_year').size().reset_index(name='count')
    yearly['year_num'] = range(1, len(yearly)+1)
    
    slope, intercept, r_value, p_value, std_err = linregress(
        yearly['year_num'], yearly['count']
    )
    
    hospital_trends.append({
        'hospital': hospital,
        'slope_per_year': round(slope, 2),
        'r_squared': round(r_value**2, 4),
        'p_value': round(p_value, 4),
        'trend': 'Increasing' if slope > 5 else ('Decreasing' if slope < -5 else 'Stable'),
        'significant': 'Yes' if p_value < 0.05 else 'No'
    })

trends_df = pd.DataFrame(hospital_trends)
print("\nTrend Analysis by Hospital:")
print(trends_df)

"""Network Centrality Analysis
Which hospitals are most central/influential in the transfer network?
In-degree centralityHow often received transfers (specialist hub?)
Out-degree centralityHow often sent transfers (referring hub?)
Betweenness centralityHow often lies on shortest path between others (bridge?)
PageRankOverall network importance"""

import networkx as nx

# Build directed network
G = nx.DiGraph()

# Add weighted edges
for (sender, receiver), count in df.groupby(['transfer_provider', 'received_provider']).size().items():
    G.add_edge(sender, receiver, weight=count)

# Calculate centrality measures
centrality_results = pd.DataFrame({
    'hospital': list(G.nodes()),
    'in_degree_centrality': list(nx.in_degree_centrality(G).values()),
    'out_degree_centrality': list(nx.out_degree_centrality(G).values()),
    'betweenness_centrality': list(nx.betweenness_centrality(G, weight='weight').values()),
    'pagerank': list(nx.pagerank(G, weight='weight').values())
})

print("\nNetwork Centrality Measures:")
print(centrality_results.sort_values('pagerank', ascending=False))

"""Visuals: Network Graph Hospital Transfer Flow"""

import networkx as nx

# Build directed network
G = nx.DiGraph()

pair_counts = df.groupby(['transfer_provider', 'received_provider']).size().reset_index(name='count')

for _, row in pair_counts.iterrows():
    G.add_edge(row['transfer_provider'], row['received_provider'], weight=row['count'])

# Layout
pos = nx.circular_layout(G)

# Edge widths scaled by weight
weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(weights)
edge_widths = [3 * w / max_weight for w in weights]

# Node sizes scaled by total activity
node_sizes = {node: df[df['transfer_provider'] == node].shape[0] + 
                    df[df['received_provider'] == node].shape[0] 
              for node in G.nodes()}

fig, ax = plt.subplots(figsize=(12, 10))

nx.draw_networkx_nodes(G, pos, 
                       node_size=[node_sizes[n] * 0.5 for n in G.nodes()],
                       node_color='#E63946', alpha=0.8, ax=ax)

nx.draw_networkx_edges(G, pos, 
                       width=edge_widths,
                       edge_color='#264653',
                       alpha=0.6,
                       arrows=True,
                       arrowsize=20,
                       ax=ax)

nx.draw_networkx_labels(G, pos, fontsize=11, fontweight='bold', ax=ax)

edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, fontsize=8, ax=ax)

ax.set_title('Hospital Transfer Network\n(Arrow thickness = Transfer volume | Node size = Total activity)', 
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/network_graph.png', dpi=300)
plt.show()

"""Visual: Network graph per transfer type"""

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
colors_by_type = {'ED_ED': '#E63946', 'ED_ADM': '#F4A261', 'ADM_ED': '#2A9D8F', 
                  'ADM_ADM': '#264653', 'OP_OP': '#E76F51'}

for idx, ttype in enumerate(df['transfer_type'].unique()):
    type_data = df[df['transfer_type'] == ttype]
    pair_counts = type_data.groupby(['transfer_provider', 'received_provider']).size().reset_index(name='count')
    
    G_type = nx.DiGraph()
    for _, row in pair_counts.iterrows():
        G_type.add_edge(row['transfer_provider'], row['received_provider'], weight=row['count'])
    
    pos = nx.circular_layout(G_type)
    weights = [G_type[u][v]['weight'] for u, v in G_type.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [4 * w / max_w for w in weights]
    
    nx.draw_networkx_nodes(G_type, pos, node_size=1500,
                          node_color=colors_by_type[ttype], alpha=0.8, ax=axes[idx])
    nx.draw_networkx_edges(G_type, pos, width=edge_widths,
                          edge_color='black', alpha=0.5,
                          arrows=True, arrowsize=20, ax=axes[idx])
    nx.draw_networkx_labels(G_type, pos, fontsize=9, fontweight='bold', ax=axes[idx])
    
    total = len(type_data)
    axes[idx].set_title(f'{ttype}\n(n={total:,})', fontsize=12, fontweight='bold',
                        color=colors_by_type[ttype])
    axes[idx].axis('off')

axes[-1].set_visible(False)
plt.suptitle('Transfer Networks by Type', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/network_by_type.png', dpi=300, bbox_inches='tight')
plt.show()


"""Heatmaps: Hospital to hospital transfer volumes"""

# Overall heatmap
heatmap_data = df.groupby(['transfer_provider', 'received_provider']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, 
            annot=True, 
            fmt='d', 
            cmap='YlOrRd',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Number of Transfers'})

ax.set_title('Hospital-to-Hospital Transfer Volume Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Receiving Hospital', fontsize=12)
ax.set_ylabel('Sending Hospital', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/heatmap_overall.png', dpi=300)
plt.show()

"""Heatmap: transfer volume per year"""

# Yearly heatmap per hospital
yearly_hospital = df.groupby(['transfer_provider', 'financial_year']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(yearly_hospital, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Number of Transfers'})

ax.set_title('Transfer Volume by Hospital and Financial Year', fontsize=14, fontweight='bold')
ax.set_xlabel('Financial Year', fontsize=12)
ax.set_ylabel('Hospital', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/heatmap_yearly.png', dpi=300)
plt.show()

"""Trend visuals: Line charts of transfer volume over time per hospital"""

yearly_by_hospital = df.groupby(['financial_year', 'transfer_provider']).size().reset_index(name='count')
hospitals = df['transfer_provider'].unique()
colors_hosp = plt.cm.Set1(np.linspace(0, 1, len(hospitals)))

fig, ax = plt.subplots(figsize=(12, 6))

for hospital, color in zip(hospitals, colors_hosp):
    data = yearly_by_hospital[yearly_by_hospital['transfer_provider'] == hospital]
    ax.plot(data['financial_year'], data['count'], 
            marker='o', linewidth=2.5, markersize=8,
            label=hospital, color=color)

ax.set_title('Yearly Transfer Trends by Hospital', fontsize=16, fontweight='bold')
ax.set_xlabel('Financial Year', fontsize=12)
ax.set_ylabel('Number of Transfers', fontsize=12)
ax.legend(title='Hospital', bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/yearly_trends_by_hospital.png', dpi=300)
plt.show()

"""Small multiples: Line charts of transfer volume over time per transfer type"""

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for idx, hospital in enumerate(hospitals):
    data = yearly_by_hospital[yearly_by_hospital['transfer_provider'] == hospital]
    
    # Scatter + regression line
    axes[idx].scatter(data['financial_year'], data['count'], 
                     s=100, color='#E63946', zorder=5)
    axes[idx].plot(data['financial_year'], data['count'], 
                  linewidth=2, color='#E63946', alpha=0.7)
    
    # Add trend line
    if len(data) > 1:
        from scipy.stats import linregress
        slope, intercept, r, p, _ = linregress(range(len(data)), data['count'])
        trend = [intercept + slope * i for i in range(len(data))]
        axes[idx].plot(data['financial_year'], trend, 
                      'b--', linewidth=1.5, alpha=0.7, label=f'R²={r**2:.2f}')
    
    axes[idx].set_title(hospital, fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Year', fontsize=9)
    axes[idx].set_ylabel('Transfers', fontsize=9)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplots
for i in range(len(hospitals), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Transfer Trends per Hospital with Regression', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/small_multiples_trends.png', dpi=300)
plt.show()

"""Visuals: Centrality measures bubble chart"""

import networkx as nx

G = nx.DiGraph()
pair_counts = df.groupby(['transfer_provider', 'received_provider']).size().reset_index(name='count')
for _, row in pair_counts.iterrows():
    G.add_edge(row['transfer_provider'], row['received_provider'], weight=row['count'])

# Calculate measures
pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight')
in_degree = nx.in_degree_centrality(G)
out_degree = nx.out_degree_centrality(G)

centrality_df = pd.DataFrame({
    'hospital': list(pagerank.keys()),
    'pagerank': list(pagerank.values()),
    'betweenness': [betweenness[n] for n in pagerank.keys()],
    'in_degree': [in_degree[n] for n in pagerank.keys()],
    'out_degree': [out_degree[n] for n in pagerank.keys()]
})

# Bubble chart
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    centrality_df['out_degree'],
    centrality_df['in_degree'],
    s=centrality_df['pagerank'] * 10000,  # Bubble size = PageRank
    c=centrality_df['betweenness'],         # Color = Betweenness
    cmap='YlOrRd',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)

# Label each hospital
for _, row in centrality_df.iterrows():
    ax.annotate(row['hospital'], 
                (row['out_degree'], row['in_degree']),
                textcoords='offset points', xytext=(10, 5),
                fontsize=10, fontweight='bold')

plt.colorbar(scatter, label='Betweenness Centrality')
ax.set_title('Hospital Network Centrality\n(Bubble size = PageRank | Color = Betweenness)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Out-Degree Centrality (Sending)', fontsize=12)
ax.set_ylabel('In-Degree Centrality (Receiving)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axvline(x=centrality_df['out_degree'].mean(), color='blue', 
           linestyle='--', alpha=0.5, label='Mean out-degree')
ax.axhline(y=centrality_df['in_degree'].mean(), color='red', 
           linestyle='--', alpha=0.5, label='Mean in-degree')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/centrality_bubble.png', dpi=300)
plt.show()

"Correlation Heatmap"

# Monthly volumes per hospital
monthly_pivot = df.groupby(
    ['transfer_provider', df['transfer_date'].dt.to_period('M')]
).size().unstack(level=0, fill_value=0)

corr_matrix = monthly_pivot.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle

sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'})

ax.set_title('Hospital Transfer Volume Correlations\n(Monthly counts)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=300)
plt.show()

"""Transfer Flow Concentration: Gini Coefficient
Is transfer activity concentrated in a few hospitals or evenly spread?
Interpretation:

Gini = 0: All 7 hospitals send/receive equally
Gini = 1: One hospital dominates all transfers
Gini > 0.4: Significant concentration"""

import numpy as np

def gini_coefficient(values):
    values = np.sort(np.array(values, dtype=float))
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

# Gini per transfer type
gini_results = []

for ttype in df['transfer_type'].unique():
    type_data = df[df['transfer_type'] == ttype]
    
    sent_counts = type_data.groupby('transfer_provider').size().values
    received_counts = type_data.groupby('received_provider').size().values
    
    gini_results.append({
        'transfer_type': ttype,
        'gini_sending': round(gini_coefficient(sent_counts), 4),
        'gini_receiving': round(gini_coefficient(received_counts), 4)
    })

gini_df = pd.DataFrame(gini_results)
print("\nGini Coefficients (0=equal, 1=concentrated):")
print(gini_df)

"""Visual: Gini Coefficient heatmap"""

# Bar chart of Gini coefficients
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sending inequality
gini_df.plot(x='transfer_type', y='gini_sending', kind='bar', 
             ax=axes[0], color='#E63946', alpha=0.8)
axes[0].set_title('Inequality in Transfer Sending\n(Gini Coefficient)', fontweight='bold')
axes[0].set_xlabel('Transfer Type')
axes[0].set_ylabel('Gini Coefficient')
axes[0].axhline(y=0.4, color='red', linestyle='--', label='High inequality (0.4)')
axes[0].legend()
axes[0].set_xticklabels(gini_df['transfer_type'], rotation=45)
axes[0].set_ylim(0, 1)

# Receiving inequality
gini_df.plot(x='transfer_type', y='gini_receiving', kind='bar', 
             ax=axes[1], color='#2A9D8F', alpha=0.8)
axes[1].set_title('Inequality in Transfer Receiving\n(Gini Coefficient)', fontweight='bold')
axes[1].set_xlabel('Transfer Type')
axes[1].set_ylabel('Gini Coefficient')
axes[1].axhline(y=0.4, color='red', linestyle='--', label='High inequality (0.4)')
axes[1].legend()
axes[1].set_xticklabels(gini_df['transfer_type'], rotation=45)
axes[1].set_ylim(0, 1)

plt.suptitle('Distribution of Transfer Activity Across 7 Hospitals', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/gini_chart.png', dpi=300)
plt.show()

"""Year-over-Year Change Per Hospital"""

# YoY for each hospital
yoy_hospital = []

for hospital in df['transfer_provider'].unique():
    yearly = df[df['transfer_provider'] == hospital].groupby('financial_year').size().reset_index(name='count')
    yearly['prev_year'] = yearly['count'].shift(1)
    yearly['pct_change'] = ((yearly['count'] - yearly['prev_year']) / yearly['prev_year'] * 100).round(2)
    yearly['hospital'] = hospital
    yoy_hospital.append(yearly)

yoy_df = pd.concat(yoy_hospital, ignore_index=True)
print("\nYear-over-Year Changes by Hospital:")
print(yoy_df[['hospital', 'financial_year', 'count', 'pct_change']])

"""Correlation Analysis: Do Hospitals Mirror Each Other?
Do hospitals' transfer volumes move together or independently?
Interpretation:

High correlation (>0.7): Hospitals respond similarly to system-wide pressures
Low correlation (<0.3): Hospitals behave independently
Useful for network resilience analysis"""

# Monthly transfers per hospital
monthly_pivot = df.groupby(
    ['transfer_provider', df['transfer_date'].dt.to_period('M')]
).size().unstack(level=0, fill_value=0)

# Correlation matrix
corr_matrix = monthly_pivot.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)




