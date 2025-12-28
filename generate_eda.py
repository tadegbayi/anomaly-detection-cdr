import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def main(csv_path='January_masked_sample.csv', out_dir='eda'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Clean numeric cols
    if 'duration' in df.columns:
        df['duration'] = df['duration'].astype(str).str.replace(',', '', regex=False)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    if 'charge' in df.columns:
        df['charge'] = pd.to_numeric(df['charge'], errors='coerce')

    # Duration distribution
    if 'duration' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['duration'].dropna(), bins=80, kde=True)
        plt.title('Duration distribution')
        plt.savefig(os.path.join(out_dir, 'eda_duration_hist.png'), bbox_inches='tight')
        plt.close()

    # Charge distribution
    if 'charge' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['charge'].dropna(), bins=80, kde=True)
        plt.title('Charge distribution')
        plt.savefig(os.path.join(out_dir, 'eda_charge_hist.png'), bbox_inches='tight')
        plt.close()

    # Boxplot: charge by top cities
    if 'city' in df.columns and 'charge' in df.columns:
        top_cities = df['city'].value_counts().nlargest(12).index
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df['city'].isin(top_cities)], x='city', y='charge')
        plt.xticks(rotation=45)
        plt.title('Charge by top cities')
        plt.savefig(os.path.join(out_dir, 'eda_charge_by_city.png'), bbox_inches='tight')
        plt.close()

    # Countplot for call_direction
    if 'call_direction' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='call_direction', order=df['call_direction'].value_counts().index)
        plt.title('Call direction counts')
        plt.savefig(os.path.join(out_dir, 'eda_call_direction_counts.png'), bbox_inches='tight')
        plt.close()

    # Correlation heatmap for numeric columns
    num_cols = [c for c in ['duration', 'charge'] if c in df.columns]
    if len(num_cols) >= 1:
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Numeric correlation')
        plt.savefig(os.path.join(out_dir, 'eda_corr.png'), bbox_inches='tight')
        plt.close()

    # PCA scatter of features (colored by duration)
    features = [f for f in ['duration', 'charge', 'city', 'destination_type', 'call_direction'] if f in df.columns]
    if features:
        X = df[features].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        c = df['duration'].fillna(0) if 'duration' in df.columns else None
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, c=c, cmap='viridis', alpha=0.6)
        if c is not None:
            plt.colorbar(sc, label='duration')
        plt.title('PCA of features colored by duration')
        plt.savefig(os.path.join(out_dir, 'eda_pca_duration.png'), bbox_inches='tight')
        plt.close()

    print(f'EDA figures saved to: {out_dir}/')


if __name__ == '__main__':
    main()
