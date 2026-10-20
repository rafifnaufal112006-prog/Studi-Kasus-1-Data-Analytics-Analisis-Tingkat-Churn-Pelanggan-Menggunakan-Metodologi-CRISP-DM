import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data_raw = {
    'Customer_ID': [101, 102, 103, 104, 104, 105, 106, 107], 
    'Monthly_Charges': [50.5, 80.0, np.nan, 65.0, 65.0, -10.0, 90.0, 45.0], 
    'CS_Tickets': [1, 4, 2, 5, 5, 0, 4, 1],
    'Churn': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data_raw)
print("--- DATA MENTAH (MESSY DATA) ---")
print(df)

df = df.drop_duplicates()

df['Monthly_Charges'] = df['Monthly_Charges'].fillna(df['Monthly_Charges'].mean())

df = df[df['Monthly_Charges'] > 0]

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

print("\n--- DATA BERSIH (READY FOR MODELING) ---")
print(df)

X = df[['Monthly_Charges', 'CS_Tickets']]
y = df['Churn']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

pelanggan_baru = pd.DataFrame([[75.0, 4]], columns=X.columns)
prediksi_churn = model.predict(pelanggan_baru)[0]

print("\n--- ANALISIS PRESKRIPTIF & KEPUTUSAN BISNIS ---")
if prediksi_churn == 1:
    print("STATUS: Pelanggan berisiko tinggi CHURN!")
    print("TINDAKAN (DECISION): Berikan promo diskon 20% dan panggil tim CS prioritas.")
else:
    print("STATUS: Pelanggan dalam kondisi AMAN.")