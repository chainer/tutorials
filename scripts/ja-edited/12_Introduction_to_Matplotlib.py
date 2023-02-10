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

# + [markdown] colab_type="text" id="MSKoKElq1kF4"
# # Matplotlib 入門
#
# グラフの描画を行う際は [Matplotlib](https://matplotlib.org/) が便利です。
# Colab では標準で Matplotlib を使ってプロットを行うと描画結果がノートブック上に表示されます。
# Matplotlib は `matplotlib.pyplot` を `plt` という別名をつけて読み込むのが一般的です。

# + colab={} colab_type="code" id="dehoAfTINPN-"
# %matplotlib inline
import matplotlib.pyplot as plt

# + [markdown] colab_type="text" id="nD6RSWpgWBpz"
# この章で用いるデータセットは前章と同じように Colab で用意されているサンプルデータを使用します。
# Colab 以外で実行する場合は、[こちら](https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv)からデータをダウンロードして、`sample_data` というディレクトリ以下に設置してください。
#
# まず、Pandas で CSV ファイルを読み込みます。

# + colab={} colab_type="code" id="jhsL4iKnjyGL"
import pandas as pd

df = pd.read_csv('sample_data/california_housing_train.csv')

df.head(5)

# + [markdown] colab_type="text" id="encTBRXeF51v"
# ## 散布図
#
# **散布図 (scatter)** は変数間の相関を視覚的に確認したり、データのばらつきや値の範囲を視覚的に確認するのに便利なものです。
# Matplotlib では与えられた配列から散布図を作成する `plt.scatter()` が用意されています。
#
# まずは、`median_income` 列のデータと `median_house_value` 列のデータをそれぞれ横軸、縦軸に取った散布図を描画してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" id="GRhhh_G2FVGW" outputId="f41d2f79-2dea-41a7-cb07-bd24fd564867"
plt.scatter(df['median_income'], df['median_house_value'])
# -

# 次に、`pupulation` 列の値と `median_house_value` 列の値をそれぞれ横軸と縦軸にとった散布図を描画します。

# + colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" id="ubHdt6jcOiKr" outputId="f0d59fb9-8948-49f5-8379-5e35d407afc3"
plt.scatter(df['population'], df['median_house_value'])

# + [markdown] colab_type="text" id="ojtzCV-ZFVPv"
# ## ヒストグラム
#
# データ中にどのような値がよく登場しているかという値ごとの頻度を確認するために使われるものに**ヒストグラム (histogram)** があります。
#
# 試しに、`median_house_values` 列の値のヒストグラムを描画してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 355} colab_type="code" id="G0icbrhoFVQ9" outputId="101a2d6a-200b-4fc0-bbfd-c984761656bc"
plt.hist(df['median_house_value'])
# -

# 上図の棒グラフ 1 つ 1 つの青い棒は、**ビン (bin)** と呼ばれ、それぞれの高さはある値の範囲に入っているサンプルの数を表します。
# ヒストグラムでは値の範囲を複数指定し、それぞれの範囲に入っているサンプルの個数を描画します。
# そのため、その値の範囲の指定を `bins` という引数を用いて行う必要があります。
# ただし、この引数はオプショナルなもので、何も与えなかった場合はビンの数が自動的に決定されます。
# この引数に整数を与えた場合は、`bins` 個のビンを値の範囲に対して等間隔に作成します。

# + colab={"base_uri": "https://localhost:8080/", "height": 532} colab_type="code" id="a9wEhPBrPnO1" outputId="48897d79-b31f-41b7-c08b-6237274590da"
# bins 引数に値を指定することで、ビンの数を指定できます
plt.hist(df['median_house_value'], bins=50)

# + [markdown] colab_type="text" id="wdyahvtpP-Lq"
# 上図から、`median_house_value` が 500,000 付近の値をとるサンプルが突出して多く存在していることが分かります。

# + [markdown] colab_type="text" id="o9SJUhlyFVS2"
# ## 箱ひげ図
#
# **箱ひげ図 (box plot)** は、値のばらつきをわかりやすく表現するための図です。
# `df.describe()` で確認できるような、いくつかの統計値をまとめて可視化するものです。
# 箱ひげ図は、**五数要約 （five-number summary）** と呼ばれる以下の統計量をまとめて表すものです。
#
# - 最小値 (minimum)
# - 第 1 四分位点 (lower quartile)
# - 中央値 (median)
# - 第 3 四分位点 (upper quartile)
# - 最大値 (maximum)
#
# 描画には、`plt.boxplot()` を用います。

# + colab={"base_uri": "https://localhost:8080/", "height": 407} colab_type="code" id="s-q3hYgXIyDu" outputId="8b9ca09c-2f6c-4175-9034-494a97f7eb3c"
plt.boxplot(df['median_house_value'])

# + colab={"base_uri": "https://localhost:8080/", "height": 532} colab_type="code" id="TbxSFRjyRgmd" outputId="6a67c378-46ef-4b9a-ef8d-d5e24c9578fa"
# 複数指定する場合は、タプルを用います
plt.boxplot((df['total_bedrooms'], df['population']))

# + [markdown] colab_type="text" id="4DyNSIl_JQyQ"
# ## 折れ線グラフ
#
# 折れ線グラフは、時系列データなどを表示する際に便利なグラフです。
# `plt.plot()` を用いて描画します。
#
# `plt.plot(y)` のように引数が 1 つの場合は、`y` の要素が縦軸の値に対応し、 横軸は要素のインデックスとなります。
#
# それでは、NumPy を用いて作成したデータを、`plt.plot()` で表示してみましょう。

# + colab={} colab_type="code" id="5sXIXrD9k_aG"
import numpy as np

# [0,10]の間を100分割して数値を返す
x = np.linspace(0, 10, 100)

# x の値にランダムノイズを加える
y = x + np.random.randn(100)

# + colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" id="RDRAWxyNJQ5p" outputId="8cc430ba-f355-4f27-805f-50e9ba0d9f6d"
plt.plot(y)

# + [markdown] colab_type="text" id="9mnZNcn_TwYN"
# `plt.plot(x, y)` のように引数を 2 つ与える場合は、`x` が横軸、`y` が縦軸に対応します。

# + colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" id="xEpVGwTaT7U2" outputId="705a3908-773c-45c7-f7d9-87f0a750bae2"
plt.plot(x, y)

# + [markdown] colab_type="text" id="dxa_v2Jf5a0m"
# ## グラフの調整
#
# Matplotlib では横軸や縦軸に文字列でラベルを指定したり、グラフの大きさの調整、また直線・曲線・点の色や大きさ、文字の色や大きさの調整など、様々な見た目に関する設定を細かく指定することができます。

# + [markdown] colab_type="text" id="Yx6XsP_QUxG-"
# ## seaborn
#
# 統計図の作成を簡単に行えるように Matplotlib をベースに作られたライブラリに [seaborn](https://seaborn.pydata.org/) というものがあります。
#
# `seaborn` パッケージは、`sns` という別名で読み込まれるのが一般的です。

# + colab={} colab_type="code" id="kvf1sgqw1EvS"
import seaborn as sns

# + [markdown] colab_type="text" id="5bPq2uG-VcL0"
# データの分布を確認する際は、Matplotlib のヒストグラムよりも使い方がシンプルかつ見やすい図を作成することができる `sns.distplot()` がおすすめです。

# + colab={"base_uri": "https://localhost:8080/", "height": 350} colab_type="code" id="bE6suI8I1E3D" outputId="7a79d6e4-2043-452d-bb82-ccac27a21cc4"
sns.distplot(df['population'])

# + [markdown] colab_type="text" id="-16KYq-AVm3Y"
# また、描画が完了するまで少し時間がかかってしまいますが、与えられたデータフレームオブジェクトの各列の全てのペアでの散布図をグリッド状に描画し、様々な変数間の相関関係を視覚的に見渡すことができる `plt.pairplot()` も便利です。

# + colab={"base_uri": "https://localhost:8080/", "height": 1643} colab_type="code" id="jbzeeKS_V1fA" outputId="5f9b8300-c9ba-44b9-b656-4692550f01c5"
sns.pairplot(df)

# + [markdown] colab_type="text" id="FYlS6tCdV1F6"
# seaborn には他にも様々な種類のグラフを描画する機能があります。
