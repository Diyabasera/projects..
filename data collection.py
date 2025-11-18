import pandas as pd

# Dataset path
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Combine both datasets
df_fake['label'] = 'FAKE'
df_real['label'] = 'REAL'
data = pd.concat([df_fake, df_real], ignore_index=True)

print(data.head())
print(data['label'].value_counts())
