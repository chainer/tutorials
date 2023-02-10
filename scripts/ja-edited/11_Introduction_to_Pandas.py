# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="jEuKb9fhjovk"
# # Pandas 入門
#
# Pandas はデータ操作によく用いられるパッケージであり、CSV などの一般的なデータ形式で保存されたデータの読み込みや、条件を指定しての一部データの抽出など、機械学習手法で取り扱うデータを整理するのに便利です。
#
# 今回は Pandas の以下の代表的な機能の使い方を説明します。  
#
# - CSV ファイルの読み書き
# - 統計量の算出
# - 並べ替え
# - データの選択
# - 条件指定による選択
# - 欠損値の除去 / 補間
# - ndarray とデータフレームを相互に変換
# - グラフの描画
#
# まず Pandas パッケージを読み込みましょう。
# `pandas` は `pd` という別名を与えて用いるのが一般的です。

# + colab={} colab_type="code" id="oOg-SRWp2OQz"
import pandas as pd

# + [markdown] colab_type="text" id="26MTeJHGqx6q"
# ## CSV ファイルの読み書き
#
# データセットは Google Colaboratory で用意されているサンプルデータを使用します。
# Google Colaboratory 以外で実行する場合は、[こちら](https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv)をダウンロードして、使用してください。
#
#

# + [markdown] colab_type="text" id="0GUo7yjIih3u"
# Pandas では CSV ファイルを読み込むための `pd.read_csv()` という関数が用意されています。
# こちらを使って CSV ファイルを読み込みます。

# + colab={} colab_type="code" id="QvqC02GJ3V0a"
# データセットの読み込み
df = pd.read_csv('sample_data/california_housing_train.csv')

# + [markdown] colab_type="text" id="k8c9lneAih3y"
# `df` という変数名は、**データフレーム (data frame)** という Pandas で中心的に用いられる**データ構造 (data structure)** を表すクラスの名前の頭文字に由来しています。
# `pd.read_csv()` 関数は、CSV ファイルの内容を `DataFrame` オブジェクトに読み込みます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="qj-5MVjXsDIY" outputId="b0263230-cd3a-45c1-e8d2-d207db627156"
# 型の確認
type(df)

# + [markdown] colab_type="text" id="KXJ2TyXAih36"
# ## DataFrame の表示

# + [markdown] colab_type="text" id="Q3BBVv61Zwst"
# `df` に読み込まれたデータの中身の確認してみましょう。
# Jupyter Notebook 上では、大きな DataFrame を表示しようとすると自動的に一部が省略されることがあります。

# + colab={"base_uri": "https://localhost:8080/", "height": 1930} colab_type="code" id="8C5F7DqQZm1y" outputId="14287b2d-a973-46d1-c9ca-e36450ae6d00"
df

# + [markdown] colab_type="text" id="1MgSOxlt3maP"
# ## 先頭の数件だけを表示
#
# データを数件のみ確認したい場合は、データフレームがもつ `df.head()` メソッドを使用します。
# `df.head()` はデフォルトで先頭から 5 件のデータを表示しますが、`df.head(3)` のように引数に表示したいデータ件数を指定すると、指定された件数だけを表示することもできます。
# それでは、`df.head()` を実行してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="JoO8FkT8s5p9" outputId="639b3b48-6e32-4f73-9f20-d727bc5c026f"
df.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="UcPEPxhr300W" outputId="0eed6f00-6935-47ff-b0c1-8c60d187fbd8"
df.head(3)

# + [markdown] colab_type="text" id="7uZd00IQ47fg"
# 特定の列を抽出したい場合は、`df` に対し、Python の辞書オブジェクトに行うように `[]` を使って取り出したい列の名前を指定します。

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="2MM8BHDt44F-" outputId="e9270663-c3a6-41b6-9dbd-62857eba1148"
df['longitude'].head(3)

# + [markdown] colab_type="text" id="7Uy36D31uibB"
# ## CSV ファイルの保存
#
# Pandas ではデータフレームオブジェクトの内容を CSV ファイルとして保存するための `df.to_csv()` というメソッドが用意されています。

# + colab={} colab_type="code" id="YpLmtcDHusQt"
df.to_csv('sample.csv')

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Qa8YM143u7Yw" outputId="7e8295ca-f99e-433d-de77-2b5a1da662ab"
# !ls sample.csv
# -

# ## データフレームの形
#
# データフレームオブジェクトの行数と列数を確認するには、`df.shape()` メソッドを用います。

# 形の確認
df.shape

# + [markdown] colab_type="text" id="7fN9CvUmvMQH"
# ## 統計量の算出
#
# データフレームには、中のデータに対し統計量を計算するためのメソッドも用意されています。
# 代表的なものを紹介します。

# + colab={"base_uri": "https://localhost:8080/", "height": 195} colab_type="code" id="zctW3yvxuAWM" outputId="1d555192-a7b2-4ec8-f3b7-b0eb67504b1f"
# 平均
df.mean()
# -

# 分散
df.var()

# + colab={"base_uri": "https://localhost:8080/", "height": 195} colab_type="code" id="qxs0qB7WuAeY" outputId="4f5f8ed4-224f-4aa1-9681-89fda20d8f4b"
# 各列の None, NaN, NaT のいずれでもない値の数
df.count()

# + [markdown] colab_type="text" id="z0TJ19egvQv7"
# ここで、データの特徴をおおまかに調べるために便利な `df.describe()` メソッドを実行してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 291} colab_type="code" id="T5kLGiCYuAbz" outputId="3de93947-60fc-4e94-96c5-b3bd407d29c3"
# データの概要
df.describe()

# + [markdown] colab_type="text" id="bBq3AXbwNzu5"
# また、もうひとつ便利なメソッドに相関係数を算出する `df.corr()` があります。
# 入力変数間や入出力間の相関係数を確認することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 321} colab_type="code" id="OnFXNN8UNoyR" outputId="1db99220-415d-4482-ddec-37f849f6aae3"
# 相関係数の算出
df.corr()

# + [markdown] colab_type="text" id="NliEdJHzvpom"
# ## 並べ替え
#
# データフレームのある列を抽出し、`df.sort_values()` メソッドを呼び出すことで値の**並べ替え (sort)** を行うことができます。
# なお、このメソッドは並べ替えが終わったあとの値でもとのデータフレーム内の値を置き換えることまでは行わず、結果を返します。
# そこで、別の変数で結果を受け取り、始めの 5 行を表示することで並べ替えが行われたことを確認してみましょう。
#
# `df.sort_values()` は、デフォルトでは**昇順 (ascending)** に並べ替えを行います。昇順とは、だんだん値が大きくなっていくように並べ替えるときの並べ方のことで、逆にだんだん値が小さくなっていくように並べ替えるときは、**降順 (descending)** に並べると言います。
#
# `df.sort_values()` は並べ替えを行いたい列の名前を `by` という引数で受け取ります。また、デフォルトでは昇順に並べ替えを行います。

# + colab={} colab_type="code" id="efmrXOKKwVmf"
# total_rooms 列の値を昇順に並べ替え
df_as = df.sort_values(by='total_rooms')

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="4-NZFmT3wVjx" outputId="bb85fa16-9260-41c9-f9f0-57d9190c29ea"
df_as.head()

# + [markdown] colab_type="text" id="JvBXPheQxExY"
# 降順に並べ替える場合は、`ascending=False` という引数の指定を行います。

# + colab={} colab_type="code" id="H5hcz3PowVgc"
# total_rooms の列の値を降順に並べ替え
df_de = df.sort_values(by='total_rooms', ascending=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="As4wF8YtwVdL" outputId="283b576f-77ff-44a1-b979-d52111377d4c"
df_de.head()

# + [markdown] colab_type="text" id="33kVroZ235bY"
# ## データの選択
#
# 着目したい要素や、行、列を選択する方法を紹介します。
# [scikit-learn 入門](https://tutorials.chainer.org/ja/09_Introduction_to_Scikit-learn.html)の章で行ったように、入力値 `x` と目標値 `t` が別の列として同じ配列に格納されている場合は、指定した列だけを取り出す操作を行って、結果を別の変数に代入する操作を行います。
# このように、特定の列の選択や列を範囲指定して選択する機能が Pandas のデータフレームにも用意されています。
#
# 今回は最後の列 `median_house_value` と、それ以外の列をそれぞれ取り出して、別々の変数に格納してみましょう。
#
# 列や行の選択を行う方法は複数あります。
# ここでは、整数インデックスを用いてデータの部分選択を行う `df.iloc[]` を紹介します。

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="e7WrVEj2zAhh" outputId="c44c0998-bcfe-49c9-b03b-cd3ceba9b7d7"
# データの確認
df.head(3)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="UFTSn1GEzF77" outputId="ea04ca7f-5a1e-4846-8776-2bb3f91af285"
# df.iloc[行, 列]
# 0 行目 longitude 列の選択
df.iloc[0, 0]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="2U1AiMzizJ0U" outputId="dc6094e3-e4a4-4534-f467-2e71836d691d"
# 1 行目 latitude 列の選択
df.iloc[1, 1]

# + [markdown] colab_type="text" id="ZPpt5xaHzacL"
# `iloc` には NumPy の ndarray の中の値を部分的に選択するのと同様のスライス表記を用いることができます。
# 負のインデックスを使い、末尾の要素からの個数を用いて位置指定を行うこともできます。

# + colab={} colab_type="code" id="HU4q_YwrzMuU"
# すべての行の、最後の列を選択
t = df.iloc[:, -1]

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="NKR5jcDp0L2B" outputId="c47f3575-f3d9-4ed2-cb99-2cd7fe9d123f"
# 先頭3件の表示
t.head(3)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="9tsxGUWo4GzU" outputId="3f5d4e7b-e61c-4b31-c0c3-eb5fa57f50be"
# 型の確認
type(t)

# + [markdown] colab_type="text" id="7TzrpxIL4F-D"
# 1 行だけ、もしくは 1 列だけ抽出した場合は、**シリーズ (series)** オブジェクトが返されます。

# + colab={} colab_type="code" id="0jN1f_720Q15"
# すべての行の、先頭の列から末尾の列のひとつ手前までを選択
x = df.iloc[:, 0:-1]
# -

# 先頭の3件の表示
x.head(3)

# + [markdown] colab_type="text" id="aiH3wkgD0Z08"
# [NumPy 入門](08_Introduction_to_NumPy_ja.ipynb)で紹介したようなスライス記法を用いる際の先頭位置の省略も行えます。

# + colab={} colab_type="code" id="9M3-hDef0RAc"
# すべての行の、先頭の列から末尾の列のひとつ手前までを選択
x = df.iloc[:, :-1]

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="yLWcPbBs0l16" outputId="0ab8e02c-7579-49dc-f8c8-948a66e0023c"
# 先頭の3件の表示
x.head(3)
# -

# 行の数、列の数両方を複数選択した場合、データフレームオブジェクトが返ります。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="r2bo3Yyp4aeN" outputId="a43559d7-7ed6-4421-8244-2e08170aa216"
# 型の確認
type(x)

# + [markdown] colab_type="text" id="zeCTdqHL7q5i"
# ## 条件指定による要素の選択
#
# 次に値に対する条件を指定してデータの選択を行う方法を紹介します。
#
# 簡単のため、まず `median_house_value` 列を選択し、返ってきたシリーズオブジェクトに対して、比較演算子を使って**各要素に対する条件**を指定し、条件を満たすかどうかを全要素に対して調べた結果を取得してみましょう。

# + colab={} colab_type="code" id="RhJW40Cz6VZj"
# median_house_value 列を選択し、全要素に対し 70000 より大きいかどうかを計算
mask = df['median_house_value'] > 70000

# + colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" id="srYmImEs6VVe" outputId="8e616a67-fe3a-4bec-fc47-b0f118f24752"
mask.head()

# + [markdown] colab_type="text" id="OQ5EFadC6nfc"
# このように、比較演算子の片方の辺にデータフレームやシリーズをおくと、指定された条件を満たすかどうかを全ての要素に対して計算することができます。
# 結果は、各要素が条件を満たすか、満たさないかを表す `True`、`False` が各要素の位置に格納されたデータフレームやシリーズとなります。
# これを**マスク (mask)** と呼ぶことがあります。
#
# そして、データフレームやシリーズも NumPy の ndarray と同様に、マスクを使って要素を選択することができます。
# 上の `mask` を `df` に `[]` を使って与えることで、指定した条件を満たす要素だけを取り出すことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="ET6N9DZi6VSV" outputId="70a034b4-0c14-4f93-d605-b006c5523ed7"
# df[mask] の先頭 5 件を表示
df[mask].head()

# + [markdown] colab_type="text" id="sRw6qcbi7U8t"
# ### 複数の条件指定による要素の選択
#
# 複数の条件を組み合わせて要素を選択することも出来ます。
# その場合は条件式を `()` でくくって用います。
# **論理和 (or)** は `|`、**論理積 (and)** は `&` を用いて表します。

# + colab={} colab_type="code" id="tYad4IqH8-TU"
# 70000 より小さい または 80000 より大きい
mask2 = (df['median_house_value'] < 70000) | (df['median_house_value'] > 80000)

# + colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" id="0QLUPiuo9IDM" outputId="19b72d99-ca79-4c74-c05a-e827f930ec88"
mask2.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="spv0H36u9mjf" outputId="aa5197a5-9356-429b-d687-f8f4fff0923c"
df[mask2].head()

# + colab={} colab_type="code" id="1El4-CLU7UPA"
# 70000 より大きい かつ 80000 より小さい
mask3 = (df['median_house_value'] > 70000) & (df['median_house_value'] < 80000)

# + colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" id="EDxNGNDp6VOl" outputId="07199cac-7d17-4b2c-9b85-534704636d09"
mask3.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="tD0geZQw9lt5" outputId="07ce72fb-b71a-472f-b98b-f46f1e010a9d"
df[mask3].head()

# + [markdown] colab_type="text" id="fbPB_J8H9ayc"
# 条件に当てはまる要素を調べる操作と、条件に当てはまる要素の選択まで、1 行にまとめて書くこともできます。

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="lNcH1ii_6N2O" outputId="d5bd3b32-2123-48e3-c889-4ca6eaea9599"
df[(df['median_house_value'] > 70000) & (df['median_house_value'] < 80000)].head()

# + [markdown] colab_type="text" id="vUdEP78l-JeI"
# ### 条件指定による要素の置換
#
# 条件を指定して選択した要素に対し、値の書き換えを行うことができます。
# 例えば、`median_house_value` 列に対していくつかの条件を別々に調べ、それぞれの条件を満たしている場合に特定の値を持つような新しい列を `df` に追加してみます。
# `median_house_value` が
#
# - 60000 より小さい場合は 0
# - 60000 以上 70000未満は 1
# - 70000 以上 80000未満は 2
# - 80000 以上は 3
#
# となる値を持つ `target` という列を追加します。
#
# まず、何も値の入っていない `target` という列を `df` に追加します。

# + colab={} colab_type="code" id="nHhpN6yT_-Ne"
# 新しい列 target を None で初期化
df['target'] = None

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="3Y8yFklUABUt" outputId="45fabe88-06b6-4804-a459-fba4e8cea7d2"
df.head()
# -

# `target` という列の全ての要素は `None` となっています。
# この値を、条件指定によって書き換えます。
#
# まず各条件に対応するマスクを作成します。

# + colab={} colab_type="code" id="l3DK9ZtI6NsT"
mask1 = df['median_house_value'] < 60000
mask2 = (df['median_house_value'] >= 60000) & (df['median_house_value'] < 70000)
mask3 = (df['median_house_value'] >= 70000) & (df['median_house_value'] < 80000)
mask4 = df['median_house_value'] >= 80000

# + [markdown] colab_type="text" id="wVdng5KDB5o1"
# 行や列を整数インデックスで選択する場合は `df.iloc[]` を使用しましたが、**列を名前で指定する**場合には `df.loc[]` を用います。
# それでは、上で計算したマスクと名前による列指定を組み合わせて、各条件を満たす行の `target` 列の値を書き換えます。

# + colab={} colab_type="code" id="KK5Hw7fv6Nou"
df.loc[mask1, 'target'] = 0
df.loc[mask2, 'target'] = 1
df.loc[mask3, 'target'] = 2
df.loc[mask4, 'target'] = 3
# -

# 結果を確認してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="0HKz4SB5_5u2" outputId="06ce6d99-0112-4243-b479-86f7b435d9da"
# 先頭から 5 番目までを表示
df.head()

# + [markdown] colab_type="text" id="vGjqaByGDShU"
# ## 欠損値の除去・補間
#
# 欠損値を含むデータの場合、一部の行の値が欠損している列に `NaN` (Not a Number)、`None`、`NaT` (Not a Time) などが含まれる場合があります。
# 欠損値への対策としては、欠損値を含む行、または列を除去するか、欠損値を特定の値で補完するという方法が考えられます。
#
# まずは、欠損値の除去の方法を紹介します。

# + colab={} colab_type="code" id="aoE0QNhuDRyQ"
# 欠損値を人為的に作成
df.iloc[0, 0] = None

# + colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="qZgx5pchDRua" outputId="24908301-5d06-4d7c-ad9b-0367afb0c25d"
# (0, 'longitude') の要素が NaN になっていることを確認
df.head(3)

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="_G5T0rQ2DRq2" outputId="8d781fab-bc5b-4cd3-b0a3-782eab114082"
# 欠損値のあるレコードを削除
df_dropna = df.dropna()

# 先頭から 3 件を表示
df_dropna.head(3)

# + [markdown] colab_type="text" id="gWpfdlT3E2UE"
# 上の結果と見比べると、`NaN` を含んでいた 0 行目のデータが取り除かれていることが分かります。
#
# 次に、平均を使った欠損値の補完を行ってみましょう。
# まずは、補完に使用する平均値の計算を行います。

# + colab={"base_uri": "https://localhost:8080/", "height": 212} colab_type="code" id="Yv9ICb3wDRm4" outputId="544bdf10-2590-4389-cff4-80a3eab92afe"
mean = df.mean()
mean
# -

# 計算した各列の値の平均が格納されている `mean` を、`df.fillna()` メソッドに渡すことで、`mean` を用いた欠損値の補完を行うことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="ucbggS34FCGu" outputId="69e897fb-6a59-4f9a-b549-acb7760d5071"
# 欠損値を mean で補完
df_fillna =  df.fillna(mean)

# 先頭から 3 件を表示
df_fillna.head(3)

# + [markdown] colab_type="text" id="e0QntVBrFN2o"
# 0 行目のデータの `longitude` 列に、`mean` の `longitude` 行の値が表示されていることが分かります。
#
# 今回は欠損値が 1 箇所にだけあるデータを用いましたが、`df.dropna()` や `df.fillna()` は、対象の全ての欠損値に対して上記のような操作を行うメソッドです。

# + [markdown] colab_type="text" id="pUfQuJt6F8IW"
# ## ndarray とデータフレームを相互に変換
#
# scikit-learn では、データフレームやシリーズをそのまま扱うことができます。
# しかし、Chainer を含む他のライブラリやフレームワークではそのままでは扱うことができない場合もあります。
# そこで、データフレームを NumPy の ndarray に変換する方法を紹介します。
#
# まず、`df` がデータフレームであることを確認します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="CIUFwSp8Gz-O" outputId="c5400086-8709-4e46-903b-27243e875d29"
type(df)
# -

# 次に、`df` の `values` という属性の型を調べてみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="I0HH67MUFCCj" outputId="73815cac-8a4a-414e-dcc1-3a10213e0c7b"
type(df.values)
# -

# NumPy の ndarray になっています。
# データフレームやシリーズは、`values` という属性に値を ndarray として格納しています。

# + colab={"base_uri": "https://localhost:8080/", "height": 248} colab_type="code" id="gVhfmvSaFB_Q" outputId="737bcb04-0041-4982-ccb4-f2bd36ca8cfa"
df.values

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="PmFD0jJ1FB7_" outputId="8cf129cf-ecb6-4e74-8f4b-67e844b3f720"
type(df['longitude'])

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Wer8w0REFB3k" outputId="cae2a740-30f3-45af-843b-096d77502efb"
type(df['longitude'].values)

# + [markdown] colab_type="text" id="ak1--qmKIM-p"
# 逆に、Python のリストや ndarray からシリーズやデータフレームを作ることもできます。
# NumPy で乱数を要素にもつ ndarray を生成し、これをデータフレームに変換してみましょう。
#
# `pd.DataFrame` のインスタンス化の際に、`data` 引数に元にしたい ndarray を与えます。

# + colab={} colab_type="code" id="LMuhjg96l1w4"
import numpy as np

# ndarray -> pd.DataFrame
df = pd.DataFrame(
    data=np.random.randn(10, 10)
)

df

# + [markdown] colab_type="text" id="XV5KiT0yKblX"
# ## グラフの描画
#
# データフレームオブジェクトから直接可視化のための機能を呼び出すことができます。
# [次の章](https://tutorials.chainer.org/ja/12_Introduction_to_Matplotlib.html)で紹介する Matplotlib というグラフ描画ライブラリを `df.plot()` 機能を用いて利用することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 364} colab_type="code" id="G23DsfMIIx7Z" outputId="f36a0f94-68b5-494c-f58a-b95263dbc365"
# グラフの描画
df.plot()

# + [markdown] colab_type="text" id="QYkTAx4IHeFA"
# Matplotlib の使い方は、次章で説明します。
