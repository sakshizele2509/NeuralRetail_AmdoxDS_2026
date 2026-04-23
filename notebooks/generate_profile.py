from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("data/silver/retail_clean.csv")

profile = ProfileReport(df, title="Retail Data Profile")
profile.to_file("reports/eda_report.html")