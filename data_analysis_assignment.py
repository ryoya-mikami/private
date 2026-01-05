import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os

# 日本語フォントの設定（環境に合わせて調整が必要な場合があります）
# Windowsの場合、MS Gothicなどが標準的です
plt.rcParams['font.family'] = 'MS Gothic'

def print_header(title):
    """見出しを分かりやすく表示する関数"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def create_sample_data(filename='sample_data.csv'):
    """テスト用のサンプルデータを作成する関数"""
    print_header("サンプルデータの作成")
    print(f"データファイルがないため、テスト用に '{filename}' を作成します...")
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'id': range(1, n_samples + 1),
        'price': np.random.normal(150, 50, n_samples) + np.random.randint(0, 100, n_samples), # 価格
        'distance_km': np.random.uniform(0.5, 20, n_samples), # 距離
        'room_type': np.random.choice(['Private', 'Shared', 'Entire home'], n_samples), # 部屋タイプ
        'review_score': np.random.randint(1, 6, n_samples), # レビュー
        'is_weekend': np.random.choice([0, 1], n_samples) # 週末かどうか
    }
    
    # 外れ値を意図的に追加
    data['price'][0] = 5000 
    
    df = pd.DataFrame(data)
    
    # 距離と価格に相関を持たせる（少しノイズを入れて）
    df['price'] = df['price'] - (df['distance_km'] * 5) + np.random.normal(0, 20, n_samples)
    df['price'] = df['price'].apply(lambda x: max(x, 10)) # 負の価格を防ぐ

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"完了: '{filename}' を作成しました。")
    return filename

def load_data():
    """データを読み込む関数"""
    print_header("1. データの読み込み")
    print("分析するデータファイル（CSVまたはTXT）を読み込みます。")
    print("※ まだデータがない場合は 'sample' と入力するとサンプルデータを作成して使用します。")
    
    while True:
        filepath = input("ファイルパスを入力してください (例: SSDSE-B-2025.csv): ").strip().strip('"').strip("'")
        
        if filepath.lower() == 'sample':
            filepath = create_sample_data()
            
        if os.path.exists(filepath):
            try:
                # 拡張子によって読み込み方を変える
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith('.txt'):
                    # タブ区切りかカンマ区切りか推定
                    try:
                        df = pd.read_csv(filepath, sep='\t')
                    except:
                        df = pd.read_csv(filepath)
                else:
                    print("警告: 対応していない拡張子ですが、CSVとして読み込みを試みます。")
                    df = pd.read_csv(filepath)
                
                print(f"\n成功: データを読み込みました！ (行数: {df.shape[0]}, 列数: {df.shape[1]})")
                print("データの先頭5行:")
                print(df.head())
                return df
            except Exception as e:
                print(f"エラー: 読み込みに失敗しました。\n詳細: {e}")
        else:
            print(f"エラー: ファイル '{filepath}' が見つかりません。もう一度入力してください。")

def handle_outliers(df):
    """外れ値を処理する関数"""
    print_header("2. 外れ値の処理")
    print("数値データの外れ値を確認し、処理します。")
    
    # 数値列のみを抽出
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("数値データが見つからないため、スキップします。")
        return df

    print("対象となる数値列:", numeric_cols)
    col_name = input(f"外れ値処理をする列名を入力してください (スキップする場合はEnter): ")
    
    if col_name in numeric_cols:
        # IQR法による外れ値検出
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]
        print(f"\n検出された外れ値の数: {len(outliers)}")
        print(f"下限: {lower_bound:.2f}, 上限: {upper_bound:.2f}")
        
        if len(outliers) > 0:
            action = input("外れ値をどうしますか？ (1: 削除する, 2: そのままにする): ")
            if action == '1':
                df_clean = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
                print(f"削除しました。残り行数: {len(df_clean)}")
                return df_clean
            else:
                print("変更せずに進みます。")
        else:
            print("外れ値はありませんでした。")
    
    return df

def basic_analysis(df):
    """基本統計量と可視化"""
    print_header("3. 基本分析と可視化")
    
    # 基本統計量
    print("\n[基本統計量]")
    print(df.describe())
    
    # 数値列の選択
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        print("可視化に必要な数値列が不足しています。")
        return

    # ヒストグラム
    print("\nヒストグラムを作成中...")
    numeric_df.hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.savefig('histogram.png')
    print("-> 'histogram.png' として保存しました。")
    plt.show() # 環境によっては表示されない場合があります
    
    # 散布図行列
    print("\n散布図行列を作成中... (時間がかかる場合があります)")
    try:
        sns.pairplot(df)
        plt.savefig('scatter_matrix.png')
        print("-> 'scatter_matrix.png' として保存しました。")
        # plt.show()
    except Exception as e:
        print(f"散布図の作成中にエラーが発生しました: {e}")

def advanced_analysis(df):
    """相関、クロス集計、ダミー変数化"""
    print_header("4. 発展的な分析")
    
    # 相関分析
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        print("\n[相関行列]")
        corr_matrix = numeric_df.corr()
        print(corr_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation.png')
        print("-> 相関ヒートマップを 'correlation.png' に保存しました。")
    
    # クロス集計
    # カテゴリカル変数（文字列など）を探す
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        print(f"\nカテゴリカル変数が見つかりました: {cat_cols}")
        if len(cat_cols) >= 1:
            target_col = cat_cols[0]
            print(f"'{target_col}' の集計:")
            print(df[target_col].value_counts())
            
            # もし2つ以上あればクロス集計
            if len(cat_cols) >= 2:
                print(f"\n'{cat_cols[0]}' と '{cat_cols[1]}' のクロス集計:")
                print(pd.crosstab(df[cat_cols[0]], df[cat_cols[1]]))
    
    # ダミー変数化 (簡単な例)
    print("\nダミー変数化を実行します（カテゴリ変数を数値化）...")
    df_dummy = pd.get_dummies(df, drop_first=True)
    print("変換後の列名:", df_dummy.columns.tolist())
    return df_dummy

def statistical_tests(df):
    """t検定と回帰分析"""
    print_header("5. 統計的検定と回帰分析")
    
    # 回帰分析のため、NaNがある行は削除しておく
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 回帰分析
    print("\n--- 回帰分析 ---")
    print("利用可能な数値列:", numeric_cols)
    
    if len(numeric_cols) < 2:
        print("回帰分析には少なくとも2つの数値列が必要です。")
        return

    y_col = input("目的変数（Y：予測したい値）の列名を入力してください: ")
    x_cols_input = input("説明変数（X：予測に使う値）の列名をカンマ区切りで入力してください (例: X1,X2): ")
    x_cols = [x.strip() for x in x_cols_input.split(',')]
    
    # 列の存在確認
    if y_col in numeric_cols and all(col in numeric_cols for col in x_cols):
        try:
            X = df[x_cols]
            y = df[y_col]
            
            # 定数項（切片）の追加
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            print(model.summary())
            
            with open('regression_results.txt', 'w') as f:
                f.write(model.summary().as_text())
            print("-> 結果を 'regression_results.txt' に保存しました。")
            
        except Exception as e:
            print(f"回帰分析エラー: {e}")
    else:
        print("エラー: 指定された列が見つかりません。")

def main():
    print(" Python 総合課題実習プログラムへようこそ ")
    print("このプログラムは、画像にある一通りの分析フロー（インポート、外れ値、可視化、回帰分析など）を実行します。")
    
    # 1. データの読み込み
    df = load_data()
    
    # 2. 外れ値処理
    df_clean = handle_outliers(df)
    
    # 3. 基本統計と可視化
    basic_analysis(df_clean)
    
    # 4. 発展分析（ダミー変数化など）
    # 回帰分析などのために、カテゴリ変数をダミー化したデータフレームを作成
    df_dummy = advanced_analysis(df_clean)
    
    # 5. 統計テストと回帰分析
    # ここではダミー変数化したデータを使うと、カテゴリ変数も回帰に含められます
    statistical_tests(df_dummy)
    
    print_header("分析終了")
    print("すべての工程が完了しました。")

if __name__ == "__main__":
    main()
