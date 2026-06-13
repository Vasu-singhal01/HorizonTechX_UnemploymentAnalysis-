# 📊 Unemployment Analysis in India — Impact of COVID-19
### Horizon TechX Data Science Internship — Task 2

---

## 📌 Project Overview
This project analyzes unemployment trends in India, with a focus on the impact of COVID-19.
It includes data cleaning, exploratory data analysis (EDA), and rich interactive visualizations
built using Plotly — all displayed in a single dashboard.

## 📁 Project Structure
```
HorizonTechX_UnemploymentAnalysis/
│
├── Unemployment_Analysis.ipynb           # Main Jupyter Notebook (run this!)
├── unemployment_analysis.py              # Python script version
├── Unemployment in India.csv             # Dataset 1
├── Unemployment_Rate_upto_11_2020.csv    # Dataset 2
├── Unemployment_Analysis_India.png       # Static Dashboard Image
└── README.md                             # Project Documentation
```

## 📊 Visualizations Included
- Monthly Unemployment Rate Trend (with COVID-19 marker)
- Pre vs During COVID Box Plot
- Top 10 States by Unemployment Rate
- Rural vs Urban Unemployment Pie Chart
- Labour Participation Rate Trend
- COVID Impact by State
- State-wise Unemployment Heatmap (Top 12 States)

## 🛠️ Libraries Used
- Python 3
- Pandas
- NumPy
- Plotly

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload both CSV dataset files
3. Paste the code from `Unemployment_Analysis.ipynb`
4. Run the cell — full interactive dashboard appears!

### Option 2 — Jupyter Notebook
1. Make sure all files are in the same folder
2. Install dependencies:
   ```
   pip install pandas numpy plotly
   ```
3. Open terminal in the folder and run:
   ```
   python -m notebook
   ```
4. Open `Unemployment_Analysis.ipynb` → Cell → Run All

## 📂 Dataset Source
Kaggle: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india

## 🔑 Key Insights
- Unemployment rate increased significantly during COVID-19 (March 2020 onwards)
- Urban areas were more impacted than rural areas
- Several states saw a sharp spike in unemployment during the lockdown period

## 👤 Author
Vasu Singhal | Bennett University | Horizon TechX Data Science Intern
