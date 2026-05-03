import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Memuat dataset Heart Disease dari UCI ML Repository."""
    print(">>> Memuat dataset...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    df.rename(columns={'num': 'target'}, inplace=True)

    # Simpan raw dataset
    df.to_csv('heart_disease_raw.csv', index=False)
    print(f"    Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("    heart_disease_raw.csv berhasil disimpan!")
    return df


def convert_target(df):
    """Konversi target ke binary (0 = tidak sakit, 1 = sakit)."""
    print(">>> Mengkonversi target ke binary...")
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f"    Distribusi target:\n{df['target'].value_counts().to_string()}")
    return df


def handle_missing_values(df):
    """Menangani missing values dengan median."""
    print(">>> Menangani missing values...")
    before = df.isnull().sum().sum()
    df['ca'].fillna(df['ca'].median(), inplace=True)
    df['thal'].fillna(df['thal'].median(), inplace=True)
    after = df.isnull().sum().sum()
    print(f"    Missing values sebelum: {before} → sesudah: {after}")
    return df


def remove_duplicates(df):
    """Menghapus data duplikat."""
    print(">>> Menghapus duplikat...")
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"    Baris sebelum: {before} → sesudah: {after} ({before - after} duplikat dihapus)")
    return df


def remove_outliers_iqr(df, columns):
    """Menghapus outlier menggunakan metode IQR."""
    print(">>> Menangani outlier (IQR)...")
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after = len(df_clean)
        print(f"    {col}: {before - after} baris dihapus")
    return df_clean


def encode_categorical(df, categorical_cols):
    """Encoding kolom kategorikal menggunakan One-Hot Encoding."""
    print(">>> Encoding kolom kategorikal...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"    Shape setelah encoding: {df_encoded.shape}")
    return df_encoded


def scale_numerical(df, numerical_cols):
    """Normalisasi kolom numerikal menggunakan StandardScaler."""
    print(">>> Scaling kolom numerikal...")
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"    Kolom yang di-scale: {numerical_cols}")
    return df


def split_and_save(df, test_size=0.2, random_state=42):
    """Train-test split dan simpan hasil preprocessing."""
    print(">>> Train-test split dan menyimpan dataset...")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Simpan file hasil preprocessing
    df.to_csv('heart_disease_preprocessing.csv', index=False)
    train_df.to_csv('heart_disease_preprocessing_train.csv', index=False)
    test_df.to_csv('heart_disease_preprocessing_test.csv', index=False)

    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("    heart_disease_preprocessing.csv berhasil disimpan!")
    print("    heart_disease_preprocessing_train.csv berhasil disimpan!")
    print("    heart_disease_preprocessing_test.csv berhasil disimpan!")

    return train_df, test_df


def preprocess():
    """Fungsi utama untuk menjalankan seluruh pipeline preprocessing."""
    print("=" * 50)
    print("  AUTOMATE PREPROCESSING - HEART DISEASE DATASET")
    print("  Ni Putu Sintia Wati")
    print("=" * 50)

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Pipeline preprocessing
    df = load_data()
    df = convert_target(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, numerical_cols)
    df = encode_categorical(df, categorical_cols)
    df = scale_numerical(df, numerical_cols)
    train_df, test_df = split_and_save(df)

    print("=" * 50)
    print("  PREPROCESSING SELESAI!")
    print(f"  Total data siap latih: {len(train_df)} baris")
    print(f"  Total data uji       : {len(test_df)} baris")
    print("=" * 50)

    return df


if __name__ == "__main__":
    preprocess()
