"""
Python 総合課題実習
授業で学んだデータインポート・外れ値処理・可視化・回帰分析を
統合的に活用し、New York City Airbnbデータを用いた実践分析を行う。

実装内容:
- データインポート・連結
- 外れ値処理
- 基本統計量
- ヒストグラム・散布図作成
- クロス集計・相関分析
- ダミー変数
- t検定
- 回帰分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  Python 総合課題実習プログラム")
print("=" * 70)
print()

# ========================================
# 1. データインポート・連結
# ========================================
print("【1. データインポート・連結】")
print("-" * 70)

def データ読み込み():
    """CSVまたはTXTファイルを読み込む"""
    print("分析するデータファイルを読み込みます。")
    print("※ データがない場合は 'sample' と入力するとサンプルデータを作成します。")
    print()
    
    while True:
        ファイルパス = input("ファイルパス: ").strip().strip('"').strip("'")
        
        # サンプルデータ作成
        if ファイルパス.lower() == 'sample':
            print("\nサンプルデータを作成中...")
            return サンプルデータ作成()
        
        # ファイル存在確認
        if not os.path.exists(ファイルパス):
            print(f"エラー: ファイル '{ファイルパス}' が見つかりません。\n")
            continue
        
        try:
            # 拡張子に応じて読み込み
            if ファイルパス.endswith('.csv'):
                df = pd.read_csv(ファイルパス, encoding='utf-8-sig')
            elif ファイルパス.endswith('.txt'):
                # タブ区切りを試す
                try:
                    df = pd.read_csv(ファイルパス, sep='\t', encoding='utf-8-sig')
                except:
                    df = pd.read_csv(ファイルパス, encoding='utf-8-sig')
            else:
                df = pd.read_csv(ファイルパス, encoding='utf-8-sig')
            
            print(f"\n✓ 読み込み成功: {df.shape[0]}行 × {df.shape[1]}列")
            print("\nデータの先頭5行:")
            print(df.head())
            print()
            return df
            
        except Exception as e:
            print(f"エラー: 読み込みに失敗しました - {e}\n")

def サンプルデータ作成():
    """テスト用のサンプルデータを作成"""
    np.random.seed(42)
    n = 300
    
    # Airbnbデータを模したサンプル
    データ = {
        'id': range(1, n + 1),
        'price': np.random.gamma(2, 50, n),  # 価格（ガンマ分布）
        'minimum_nights': np.random.choice([1, 2, 3, 7, 30], n),
        'number_of_reviews': np.random.poisson(20, n),
        'reviews_per_month': np.random.uniform(0, 5, n),
        'availability_365': np.random.randint(0, 366, n),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 
                                      n, p=[0.5, 0.4, 0.1]),
        'neighbourhood_group': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx'], 
                                                n, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(データ)
    
    # 外れ値を意図的に追加
    df.loc[0, 'price'] = 10000
    df.loc[1, 'price'] = 8500
    
    # 相関を持たせる（レビュー数が多いほど価格が高い傾向）
    df['price'] = df['price'] + df['number_of_reviews'] * 2 + np.random.normal(0, 30, n)
    df['price'] = df['price'].clip(lower=10)
    
    ファイル名 = 'sample_airbnb_data.csv'
    df.to_csv(ファイル名, index=False, encoding='utf-8-sig')
    print(f"✓ サンプルデータ '{ファイル名}' を作成しました。")
    print(f"  {df.shape[0]}行 × {df.shape[1]}列")
    print("\nデータの先頭5行:")
    print(df.head())
    print()
    return df

# データ読み込み実行
df = データ読み込み()

# ========================================
# 2. 外れ値処理
# ========================================
print("\n【2. 外れ値処理】")
print("-" * 70)

def 外れ値処理(df):
    """IQR法による外れ値の検出と処理"""
    数値列 = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not 数値列:
        print("数値データが見つかりません。スキップします。\n")
        return df
    
    print("数値列:", ', '.join(数値列))
    print()
    
    対象列 = input("外れ値処理する列名を入力 (スキップする場合はEnter): ").strip()
    
    if 対象列 not in 数値列:
        print("処理をスキップします。\n")
        return df
    
    # IQR法
    Q1 = df[対象列].quantile(0.25)
    Q3 = df[対象列].quantile(0.75)
    IQR = Q3 - Q1
    下限 = Q1 - 1.5 * IQR
    上限 = Q3 + 1.5 * IQR
    
    外れ値 = df[(df[対象列] < 下限) | (df[対象列] > 上限)]
    
    print(f"\nIQR法による外れ値検出結果:")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  下限: {下限:.2f}")
    print(f"  上限: {上限:.2f}")
    print(f"  外れ値の数: {len(外れ値)}個")
    
    if len(外れ値) > 0:
        print(f"\n外れ値の例:")
        print(外れ値[[対象列]].head())
        
        処理方法 = input("\n外れ値を削除しますか？ (y/n): ").strip().lower()
        if 処理方法 == 'y':
            df_clean = df[(df[対象列] >= 下限) & (df[対象列] <= 上限)].copy()
            print(f"✓ 外れ値を削除しました。残り: {len(df_clean)}行")
            print()
            return df_clean
    
    print("データはそのまま使用します。\n")
    return df

df = 外れ値処理(df)

# ========================================
# 3. 基本統計量
# ========================================
print("\n【3. 基本統計量】")
print("-" * 70)
print(df.describe())
print()

# ========================================
# 4. ヒストグラム・散布図作成
# ========================================
print("\n【4. ヒストグラム・散布図作成】")
print("-" * 70)

数値df = df.select_dtypes(include=[np.number])

if len(数値df.columns) > 0:
    # ヒストグラム
    print("ヒストグラムを作成中...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(数値df.columns[:6]):
        数値df[col].hist(bins=30, ax=axes[i], edgecolor='black')
        axes[i].set_title(f'{col} の分布', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('度数')
    
    # 余ったサブプロットを非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('ヒストグラム.png', dpi=150, bbox_inches='tight')
    print("✓ 'ヒストグラム.png' を保存しました。")
    plt.close()
    
    # 散布図
    if len(数値df.columns) >= 2:
        print("散布図を作成中...")
        plt.figure(figsize=(10, 8))
        sns.pairplot(df.select_dtypes(include=[np.number]).iloc[:, :4])
        plt.savefig('散布図.png', dpi=150, bbox_inches='tight')
        print("✓ '散布図.png' を保存しました。")
        plt.close()

print()

# ========================================
# 5. クロス集計・相関分析
# ========================================
print("\n【5. クロス集計・相関分析】")
print("-" * 70)

# 相関分析
if len(数値df.columns) > 1:
    print("相関行列:")
    相関行列 = 数値df.corr()
    print(相関行列)
    print()
    
    # 相関ヒートマップ
    plt.figure(figsize=(10, 8))
    sns.heatmap(相関行列, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('相関行列ヒートマップ', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('相関分析.png', dpi=150, bbox_inches='tight')
    print("✓ '相関分析.png' を保存しました。")
    plt.close()

# クロス集計
カテゴリ列 = df.select_dtypes(exclude=[np.number]).columns.tolist()
if len(カテゴリ列) >= 2:
    print(f"\nクロス集計 ({カテゴリ列[0]} × {カテゴリ列[1]}):")
    クロス表 = pd.crosstab(df[カテゴリ列[0]], df[カテゴリ列[1]])
    print(クロス表)
    print()

# ========================================
# 6. ダミー変数
# ========================================
print("\n【6. ダミー変数】")
print("-" * 70)

if カテゴリ列:
    print(f"カテゴリ変数を数値化します: {', '.join(カテゴリ列)}")
    df_ダミー = pd.get_dummies(df, columns=カテゴリ列, drop_first=True)
    print(f"✓ ダミー変数化完了。列数: {df.shape[1]} → {df_ダミー.shape[1]}")
    print("新しい列名の例:", df_ダミー.columns.tolist()[:10])
    print()
else:
    df_ダミー = df.copy()
    print("カテゴリ変数がありません。\n")

# ========================================
# 7. t検定
# ========================================
print("\n【7. t検定】")
print("-" * 70)

if カテゴリ列 and len(数値df.columns) > 0:
    print("2群間の平均値の差を検定します。")
    
    # 最初のカテゴリ変数の最初の2グループを使用
    カテゴリ = カテゴリ列[0]
    グループ = df[カテゴリ].unique()[:2]
    
    if len(グループ) >= 2 and len(数値df.columns) > 0:
        数値変数 = 数値df.columns[0]
        
        グループ1 = df[df[カテゴリ] == グループ[0]][数値変数].dropna()
        グループ2 = df[df[カテゴリ] == グループ[1]][数値変数].dropna()
        
        t統計量, p値 = stats.ttest_ind(グループ1, グループ2)
        
        print(f"\n変数: {数値変数}")
        print(f"グループ1 ({グループ[0]}): 平均={グループ1.mean():.2f}, n={len(グループ1)}")
        print(f"グループ2 ({グループ[1]}): 平均={グループ2.mean():.2f}, n={len(グループ2)}")
        print(f"\nt統計量: {t統計量:.4f}")
        print(f"p値: {p値:.4f}")
        
        if p値 < 0.05:
            print("→ 有意差あり (p < 0.05)")
        else:
            print("→ 有意差なし (p >= 0.05)")
        print()

# ========================================
# 8. 回帰分析
# ========================================
print("\n【8. 回帰分析】")
print("-" * 70)

数値列リスト = df_ダミー.select_dtypes(include=[np.number]).columns.tolist()

if len(数値列リスト) >= 2:
    print("利用可能な数値列:", ', '.join(数値列リスト[:10]))
    print()
    
    目的変数 = input("目的変数 (Y) の列名: ").strip()
    説明変数入力 = input("説明変数 (X) の列名 (カンマ区切り): ").strip()
    説明変数 = [x.strip() for x in 説明変数入力.split(',')]
    
    if 目的変数 in 数値列リスト and all(x in 数値列リスト for x in 説明変数):
        try:
            # データ準備
            分析df = df_ダミー[[目的変数] + 説明変数].dropna()
            
            X = 分析df[説明変数]
            y = 分析df[目的変数]
            
            # 定数項追加
            X = sm.add_constant(X)
            
            # OLS回帰
            モデル = sm.OLS(y, X).fit()
            
            print("\n" + "=" * 70)
            print("回帰分析結果")
            print("=" * 70)
            print(モデル.summary())
            
            # 結果をファイルに保存
            with open('回帰分析結果.txt', 'w', encoding='utf-8') as f:
                f.write(モデル.summary().as_text())
            
            print("\n✓ '回帰分析結果.txt' に保存しました。")
            
        except Exception as e:
            print(f"エラー: {e}")
    else:
        print("エラー: 指定された列が見つかりません。")
else:
    print("回帰分析に必要な数値列が不足しています。")

print("\n" + "=" * 70)
print("  すべての分析が完了しました")
print("=" * 70)
