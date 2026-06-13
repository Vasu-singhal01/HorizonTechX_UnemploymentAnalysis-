# ============================================================
#  Unemployment Analysis in India — Impact of COVID-19
#  Horizon TechX Data Science Internship — Task 2
#  Dataset Source: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ─────────────────────────────────────────
df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# ── 2. Data Cleaning ─────────────────────────────────────────
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), dayfirst=True)
df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), dayfirst=True)

# Combine both datasets
df_all = pd.concat([df1, df2], ignore_index=True)
df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True, errors='coerce')
df_all = df_all.dropna(subset=['Date'])

# Feature Engineering
df_all['Month'] = df_all['Date'].dt.to_period('M')
df_all['COVID'] = df_all['Date'].apply(
    lambda x: 'During COVID' if x >= pd.Timestamp('2020-03-01') else 'Pre COVID'
)

col_ur  = 'Estimated Unemployment Rate (%)'
col_emp = 'Estimated Employed'
col_lpr = 'Estimated Labour Participation Rate (%)'

# ── 3. Basic EDA ─────────────────────────────────────────────
print("=" * 55)
print("  UNEMPLOYMENT ANALYSIS IN INDIA — EDA SUMMARY")
print("=" * 55)
print(f"\nTotal Records     : {len(df_all)}")
print(f"States Covered    : {df_all['Region'].nunique()}")
print(f"Date Range        : {df_all['Date'].min().date()} → {df_all['Date'].max().date()}")
print(f"\nAvg Unemployment (Pre-COVID)    : {df_all[df_all['COVID']=='Pre COVID'][col_ur].mean():.2f}%")
print(f"Avg Unemployment (During COVID) : {df_all[df_all['COVID']=='During COVID'][col_ur].mean():.2f}%")
peak_idx = df_all[col_ur].idxmax()
print(f"Peak Unemployment Rate          : {df_all.loc[peak_idx, col_ur]:.2f}% ({df_all.loc[peak_idx, 'Region']})")
print("\nTop 5 States by Avg Unemployment:")
print(df_all.groupby('Region')[col_ur].mean().sort_values(ascending=False).head(5).to_string())

# ── 4. Visualization ─────────────────────────────────────────
BG      = '#0d1117'
CARD    = '#161b22'
ACCENT1 = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
ACCENT4 = '#d2a8ff'
ACCENT5 = '#ffa657'
TEXT    = '#e6edf3'
SUBTEXT = '#8b949e'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   CARD,
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  TEXT,
    'xtick.color':      SUBTEXT,
    'ytick.color':      SUBTEXT,
    'text.color':       TEXT,
    'grid.color':       '#21262d',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
})

fig = plt.figure(figsize=(22, 28), facecolor=BG)
gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35)
fig.subplots_adjust(top=0.93, bottom=0.04, left=0.06, right=0.97)

# Title Banner
ax_title = fig.add_axes([0, 0.945, 1, 0.055])
ax_title.set_facecolor('#161b22')
ax_title.axis('off')
ax_title.text(0.5, 0.72, '📊  Unemployment Analysis in India — Impact of COVID-19',
              ha='center', va='center', fontsize=22, fontweight='bold', color=TEXT,
              transform=ax_title.transAxes)
ax_title.text(0.5, 0.18, 'Data Science Internship Project  |  Horizon TechX  |  Dataset: Kaggle',
              ha='center', va='center', fontsize=11, color=SUBTEXT, transform=ax_title.transAxes)

# KPI Cards
kpi_ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
pre    = df_all[df_all['COVID']=='Pre COVID'][col_ur].mean()
dur    = df_all[df_all['COVID']=='During COVID'][col_ur].mean()
peak   = df_all[col_ur].max()
peak_s = df_all.loc[df_all[col_ur].idxmax(), 'Region']
kpis   = [
    ('Avg Unemployment\n(Pre-COVID)',    f'{pre:.1f}%',           ACCENT3),
    ('Avg Unemployment\n(During COVID)', f'{dur:.1f}%',           ACCENT2),
    ('Peak Unemployment\nRate',          f'{peak:.1f}%\n({peak_s})', ACCENT4),
]
for ax, (label, val, color) in zip(kpi_ax, kpis):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, val,   ha='center', va='center', fontsize=22, fontweight='bold', color=color,  transform=ax.transAxes)
    ax.text(0.5, 0.18, label, ha='center', va='center', fontsize=11, color=SUBTEXT, transform=ax.transAxes)

# Chart 1 — Monthly Trend
ax1      = fig.add_subplot(gs[1, :2])
monthly  = df_all.groupby('Month')[col_ur].mean().reset_index()
monthly['Date'] = monthly['Month'].dt.to_timestamp()
monthly  = monthly.sort_values('Date')
covid_start = pd.Timestamp('2020-03-01')
ax1.fill_between(monthly['Date'], monthly[col_ur], alpha=0.25, color=ACCENT1)
ax1.plot(monthly['Date'], monthly[col_ur], color=ACCENT1, linewidth=2.5, marker='o', markersize=4)
ax1.axvline(covid_start, color=ACCENT2, linestyle='--', linewidth=1.8, label='COVID-19 Onset (Mar 2020)')
ax1.fill_betweenx([0, monthly[col_ur].max()+5], covid_start, monthly['Date'].max(), alpha=0.07, color=ACCENT2)
ax1.set_title('Monthly Average Unemployment Rate Trend', fontsize=13, fontweight='bold', pad=10, color=TEXT)
ax1.set_ylabel('Unemployment Rate (%)', fontsize=10)
ax1.set_xlabel('Month', fontsize=10)
ax1.legend(fontsize=9, facecolor=CARD, edgecolor='#30363d')
ax1.grid(True, axis='y')

# Chart 2 — Box Plot Pre vs During COVID
ax2 = fig.add_subplot(gs[1, 2])
bp  = ax2.boxplot([df_all[df_all['COVID']=='Pre COVID'][col_ur],
                   df_all[df_all['COVID']=='During COVID'][col_ur]],
                  patch_artist=True,
                  medianprops=dict(color='white', linewidth=2),
                  whiskerprops=dict(color=SUBTEXT),
                  capprops=dict(color=SUBTEXT),
                  flierprops=dict(marker='o', color=SUBTEXT, markersize=3))
bp['boxes'][0].set_facecolor(ACCENT3+'88'); bp['boxes'][0].set_edgecolor(ACCENT3)
bp['boxes'][1].set_facecolor(ACCENT2+'88'); bp['boxes'][1].set_edgecolor(ACCENT2)
ax2.set_xticklabels(['Pre COVID', 'During COVID'], fontsize=9)
ax2.set_title('Unemployment Distribution\nPre vs During COVID', fontsize=12, fontweight='bold', pad=8, color=TEXT)
ax2.set_ylabel('Unemployment Rate (%)', fontsize=9)
ax2.grid(True, axis='y')

# Chart 3 — Top 10 States
ax3        = fig.add_subplot(gs[2, :2])
state_avg  = df_all.groupby('Region')[col_ur].mean().sort_values(ascending=False).head(10)
colors_bar = [ACCENT2 if i < 3 else ACCENT1 for i in range(len(state_avg))]
bars       = ax3.barh(state_avg.index[::-1], state_avg.values[::-1], color=colors_bar[::-1], edgecolor='none', height=0.65)
for bar, val in zip(bars, state_avg.values[::-1]):
    ax3.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9, color=TEXT)
ax3.set_title('Top 10 States by Average Unemployment Rate', fontsize=13, fontweight='bold', pad=10, color=TEXT)
ax3.set_xlabel('Average Unemployment Rate (%)', fontsize=10)
ax3.grid(True, axis='x')

# Chart 4 — Rural vs Urban Pie
ax4 = fig.add_subplot(gs[2, 2])
if 'Area' in df1.columns:
    area_data = df1.groupby('Area')[col_ur].mean()
    wedges, texts, autotexts = ax4.pie(
        area_data.values, labels=area_data.index, autopct='%1.1f%%',
        colors=[ACCENT1, ACCENT5], textprops={'color': TEXT, 'fontsize': 11},
        wedgeprops={'edgecolor': BG, 'linewidth': 2}, startangle=90)
    for at in autotexts:
        at.set_color(BG); at.set_fontweight('bold')
    ax4.set_title('Rural vs Urban\nUnemployment Share', fontsize=12, fontweight='bold', pad=8, color=TEXT)

# Chart 5 — Labour Participation Rate
ax5      = fig.add_subplot(gs[3, :2])
lpr_m    = df_all.groupby('Month')[col_lpr].mean().reset_index()
lpr_m['Date'] = lpr_m['Month'].dt.to_timestamp()
lpr_m    = lpr_m.sort_values('Date')
ax5.fill_between(lpr_m['Date'], lpr_m[col_lpr], alpha=0.2, color=ACCENT3)
ax5.plot(lpr_m['Date'], lpr_m[col_lpr], color=ACCENT3, linewidth=2.5, marker='s', markersize=4)
ax5.axvline(covid_start, color=ACCENT2, linestyle='--', linewidth=1.8, label='COVID-19 Onset')
ax5.set_title('Labour Participation Rate Trend Over Time', fontsize=13, fontweight='bold', pad=10, color=TEXT)
ax5.set_ylabel('Labour Participation Rate (%)', fontsize=10)
ax5.set_xlabel('Month', fontsize=10)
ax5.legend(fontsize=9, facecolor=CARD, edgecolor='#30363d')
ax5.grid(True, axis='y')

# Chart 6 — COVID Impact by State
ax6    = fig.add_subplot(gs[3, 2])
impact = df_all.groupby(['Region','COVID'])[col_ur].mean().unstack()
if 'Pre COVID' in impact.columns and 'During COVID' in impact.columns:
    impact['change']   = impact['During COVID'] - impact['Pre COVID']
    top_impact         = impact['change'].dropna().sort_values(ascending=False).head(8)
    bar_colors_impact  = [ACCENT2 if v > 0 else ACCENT3 for v in top_impact.values]
    ax6.barh(top_impact.index[::-1], top_impact.values[::-1], color=bar_colors_impact[::-1], height=0.6)
    ax6.axvline(0, color=SUBTEXT, linewidth=1)
    ax6.set_title('COVID Impact by State\n(Change in Unemployment %)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
    ax6.set_xlabel('Change (%)', fontsize=9)
    ax6.grid(True, axis='x')

# Chart 7 — Heatmap
ax7      = fig.add_subplot(gs[4, :])
heat_data = df_all.groupby(['Region','Month'])[col_ur].mean().unstack()
heat_data = heat_data.loc[heat_data.mean(axis=1).sort_values(ascending=False).head(12).index]
heat_data.columns = [str(c) for c in heat_data.columns]
sns.heatmap(heat_data, ax=ax7, cmap='RdYlGn_r', linewidths=0.3,
            linecolor='#0d1117', annot=False,
            cbar_kws={'label': 'Unemployment Rate (%)', 'shrink': 0.8})
ax7.set_title('State-wise Unemployment Rate Heatmap (Top 12 States)', fontsize=13, fontweight='bold', pad=10, color=TEXT)
ax7.set_xlabel('Month-Year', fontsize=10)
ax7.set_ylabel('State', fontsize=10)
ax7.tick_params(axis='x', rotation=45, labelsize=7)
ax7.tick_params(axis='y', labelsize=9)
cbar = ax7.collections[0].colorbar
cbar.ax.yaxis.label.set_color(TEXT)
cbar.ax.tick_params(colors=TEXT)

plt.savefig('Unemployment_Analysis_India.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.show()
print("\n✅ Analysis complete! Dashboard saved as 'Unemployment_Analysis_India.png'")
