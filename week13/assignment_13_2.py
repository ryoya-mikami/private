import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定 (MS Gothic)
plt.rcParams['font.family'] = 'MS Gothic'

def main():
    # 1. データの準備 (2021年度 一人当たり県民所得 - 内閣府)
    data = {
        '都道府県': [
            '東京都', '愛知県', '茨城県', '静岡県', '栃木県', '富山県', '福井県', '山梨県', '徳島県', '神奈川県',
            '群馬県', '広島県', '滋賀県', '三重県', '岐阜県', '和歌山県', '千葉県', '大阪府', '埼玉県', '京都府',
            '兵庫県', '石川県', '山口県', '長野県', '福島県', '新潟県', '島根県', '宮城県', '山形県', '青森県',
            '香川県', '北海道', '長崎県', '岡山県', '福岡県', '大分県', '佐賀県', '秋田県', '岩手県', '愛媛県',
            '高知県', '熊本県', '鹿児島県', '奈良県', '鳥取県', '宮崎県', '沖縄県'
        ],
        '一人当たり県民所得(千円)': [
            5761, 3597, 3438, 3314, 3307, 3291, 3263, 3243, 3202, 3199,
            3187, 3179, 3161, 3111, 3092, 3084, 3059, 3051, 3049, 3026,
            2997, 2963, 2960, 2949, 2921, 2919, 2909, 2865, 2861, 2858,
            2851, 2811, 2752, 2743, 2733, 2709, 2689, 2689, 2685, 2670,
            2653, 2608, 2572, 2549, 2507, 2393, 2258
        ]
    }
    
    # DataFrame作成
    df = pd.DataFrame(data)
    
    # 2. 分析: ローレンツ曲線とジニ係数の計算
    # 所得の低い順にソート
    df_sorted = df.sort_values('一人当たり県民所得(千円)').reset_index(drop=True)
    
    # 人口は簡略化のため各都道府県で等しいと仮定 (別解として人口データを入れることも可能だが、設問の趣旨は「例」の提案なので簡易版とする)
    # ※厳密には各県の人口で重み付けすべきですが、基本的な傾向を見るため「県単位の格差」として扱います。
    n = len(df_sorted)
    
    # 累積相対度数 (人口: 横軸) - 0から1まで等間隔
    cum_pop = np.arange(n + 1) / n
    
    # 累積所得 (所得: 縦軸)
    incomes = df_sorted['一人当たり県民所得(千円)'].values
    cum_income_amount = np.concatenate([[0], np.cumsum(incomes)])
    cum_income_ratio = cum_income_amount / cum_income_amount[-1]
    
    # ジニ係数の計算 (台形公式)
    # Gini = 1 - 2 * (ローレンツ曲線の下の面積)
    # 面積を明示的に計算（np.trapz を使わない安全な実装）
    area = 0.0
    for i in range(1, len(cum_pop)):
        area += 0.5 * (cum_income_ratio[i] + cum_income_ratio[i-1]) * (cum_pop[i] - cum_pop[i-1])
    gini_coefficient = 1 - 2 * area
    
    print(f"ジニ係数: {gini_coefficient:.4f}")
    
    # 3. 可視化: ローレンツ曲線
    plt.figure(figsize=(10, 8))
    plt.plot(cum_pop, cum_income_ratio, label='ローレンツ曲線', marker='o', markersize=4)
    plt.plot([0, 1], [0, 1], 'r--', label='完全平等線')
    plt.fill_between(cum_pop, cum_pop, cum_income_ratio, alpha=0.1, color='orange', label='不平等の大きさ')
    
    plt.title(f'都道府県別 一人当たり県民所得のローレンツ曲線 (2021年度)\nジニ係数: {gini_coefficient:.3f}', fontsize=16)
    plt.xlabel('累積都道府県比率 (所得の低い順)', fontsize=12)
    plt.ylabel('累積所得比率', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.5)
    
    # グラフを保存
    plt.savefig('lorentz_curve_income.png')
    print("グラフを 'lorentz_curve_income.png' に保存しました。")

    # 4. 階級別分布表 (四分位数)
    print("\n[階級別分布表 (四分位数)]")
    quartiles = pd.qcut(df_sorted['一人当たり県民所得(千円)'], 4, labels=['第1四分位(低)', '第2四分位', '第3四分位', '第4四分位(高)'])
    df_sorted['階級'] = quartiles
    
    summary_table = df_sorted.groupby('階級', observed=False)['一人当たり県民所得(千円)'].agg(['count', 'min', 'max', 'mean'])
    print(summary_table)
    
    # 結果のテキスト出力
    with open('analysis_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"ジニ係数: {gini_coefficient:.4f}\n\n")
        f.write("[階級別分布表]\n")
        f.write(summary_table.to_string())
        f.write("\n\n[上位5都道府県]\n")
        f.write(df_sorted.tail(5)[::-1].to_string())
        f.write("\n\n[下位5都道府県]\n")
        f.write(df_sorted.head(5).to_string())

if __name__ == "__main__":
    main()
