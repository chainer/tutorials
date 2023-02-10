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

# + [markdown] colab_type="text" id="EFXvywOA1eV9"
# # scikit-learn 入門
#
# scikit-learn は Python のオープンソース機械学習ライブラリです。  
# 様々な機械学習の手法が統一的なインターフェースで利用できるようになっています。
# scikit-learn では NumPy の ndarray でデータやパラメータを取り扱うため、他のライブラリとの連携もしやすくなっています。
#
# 本章では、この scikit-learn というライブラリを用いて、**データを使ってモデルを訓練し、評価するという一連の流れを解説**し、Chainer を使ったディープラーニングの解説に入る前に、共通する重要な項目について説明します。
#
# 機械学習の様々な手法を用いる際には、**データを使ってモデルを訓練する**までに、以下の **5 つのステップ**がよく共通して現れます。
#
# - Step 1：**データセットの準備**
# - Step 2：**モデルを決める**
# - Step 3：**目的関数を決める**
# - Step 4：**最適化手法を選択する**
# - Step 5：**モデルを訓練する**
#
# 前章では、**ステップ 2** → **ステップ 3** → **ステップ 4 & 5** という **3 ステップ**に分けて説明を行いましたが、この 5 つのステップに分けて考える方法は、後の章で解説する **Chainer を用いたニューラルネットワークの訓練においても共通しています。**
# また、上記の 5 つが完了した後には、通常、訓練済みモデルによるテストデータを用いた精度検証を行いますが、この点も共通しています。
#
# 本章では、これらの 5 つのステップ + テストデータでの精度検証までを、scikit-learn の機能を使って簡潔に紹介します。

# + [markdown] colab_type="text" id="IBsdvRtx1eV_"
# ## scikit-learn を用いた重回帰分析
#
# 前章で NumPy を用いて実装した重回帰分析を、scikit-learn を使ってより大きなデータセットに対し適用してみましょう。
#
# ### Step 1：データセットの準備
#
# 本章では、前章までのような人工データではなく、米国ボストンの 506 の地域ごとの住環境の情報などと家賃の中央値の情報を収集して作られた Boston house prices dataset というデータセットを使用します。
#
# このデータセットには 506 件のサンプルが含まれており、各サンプルは以下の情報を持っています。
#
# | 属性名 | 説明 |
# |:--|:--|
# | CRIM | 人口 1 人あたりの犯罪発生率 |
# | ZN | 25,000 平方フィート以上の住宅区画が占める割合 |
# | INDUS | 小売業以外の商業が占める面積の割合 |
# | CHAS | チャールズ川の川沿いかどうか (0 or 1) |
# | NOX | 窒素酸化物の濃度 |
# | RM | 住居の平均部屋数 |
# | AGE | 1940 年より前に建てられた持ち主が住んでいる物件の割合 |
# | DIS | 5 つのボストン雇用施設からの重み付き距離 |
# | RAD | 環状高速道路へのアクセシビリティ指標 |
# | TAX | $10,000 あたりの固定資産税率 |
# | PTRATIO | 町ごとにみた教師 1 人あたりの生徒数 |
# | B | 町ごとにみた黒人の比率を Bk としたときの (Bk - 0.63)^2 の値 |
# | LSTAT | 給与の低い職業に従事する人口の割合 |
# | MEDV | 物件価格の中央値 |
#
# このデータセットを用いて、最後の MEDV 以外の 13 個の指標から、MEDV を予測する回帰問題に取り組んでみましょう。
# このデータセットは、scikit-learn の `load_boston()` という関数を呼び出すことで読み込むことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" id="v7s7ivnO1eWA" outputId="3134bcd1-1e5b-4a65-8f20-3b0c201e7c43"
from sklearn.datasets import load_boston

dataset = load_boston()
# -

# 読み込んだデータセットは、`data` という属性と `target` という属性を持っており、それぞれに入力値と目標値を並べた ndarray が格納されています。
# これらを取り出して、それぞれ `x` と `t` という変数に格納しておきましょう。

x = dataset.data
t = dataset.target

# 入力値が格納されている `x` は、506 個の 13 次元ベクトルを並べたものになっています。
# 形を確認してみましょう。

x.shape

# 一方 `t` は、各データ点ごとに 1 つの値を持つため、506 次元のベクトルになっています。
# 形を確認してみましょう。

t.shape

# #### データセットの分割
#
# ここで、まずこのデータセットを 2 つに分割します。
# それは、モデルの訓練に用いるためのデータと、訓練後のモデルのパフォーマンスをテストするために用いるデータは、異なるものになっている必要があるためです。
# これは、[機械学習に使われる数学](https://tutorials.chainer.org/ja/03_Basic_Math_for_Machine_Learning.html)の章で少しだけ触れた汎化性能というものに関わる重要なことです。
#
# ここで、例え話を使ってなぜデータセットを分割する必要があるかを説明します。
# 例えば、大学受験の準備のために 10 年分の過去問を購入し、一部を**勉強のため**に、一部を**勉強の成果をはかる**ために使用したいとします。
# 10 年分という限られた数の問題を使って、結果にある程度の信頼のおけるような方法で実力をチェックするには、下記の 2 つのうちどちらの方法がより良いでしょうか。
#
# - 10 年分の過去問全てを使って勉強したあと、もう一度同じ問題を使って実力をはかる
# - 5 年分の過去問だけを使って勉強し、残りの 5 年分の未だ見たことがない問題を使って実力をはかる
#
# 一度勉強した問題を再び解くことができると確認できても、大学受験の当日に未知の問題が出たときにどの程度対処できるかを事前にチェックするには不十分です。
# よって、後者のような方法で数限られた問題を活用する方が、本当の実力をはかるには有効でしょう。
#
# これは機械学習におけるモデルの訓練と検証でも同様に言えることです。
# **実力をつける**ための勉強に使うデータの集まりを、**訓練用データセット (training dataset)** といい、**実力をはかる**ために使うデータの集まりを、**テスト用データセット (test dataset)** と言います。
# このとき、訓練用データセットとテスト用データセットに含まれるデータの間には、**重複がないようにします。**
#
#
#
# 早速、さきほど用意した `x` と `t` を、訓練用データセットとテスト用データセットに分割しましょう。
# どのように分けるかには色々な方法がありますが、単純に全体の何割かを訓練用データセットとし、残りをテスト用データセットとする、といった分割を行う方法は**ホールドアウト法 (holdout method)** と呼ばれます。
# scikit-learn では、データセットから指定された割合（もしくは個数）のデータをランダムに抽出して訓練用データセットを作成し、残りをテスト用データセットとする処理を行う関数が提供されています。

# +
# データセットを分割する関数の読み込み
from sklearn.model_selection import train_test_split

# 訓練用データセットとテスト用データセットへの分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
# -

# ここで、`train_test_split()` の `test_size` という引数に 0.3 を与えています。
# これはテスト用データセットを全体の 30% のデータを用いて作成することを意味しています。
# 自動的に残りの 70% は訓練用データセットとなります。
# 上のコードは全サンプルの中から**ランダムに** 70% を訓練データとして抽出し、残った 30% をテストデータとして返します。
#
#
# 例えば、データセット中のサンプルが、目標値が 1 のサンプルが 10 個、2 のサンプルが 8 個、3 のサンプルが 12個…というように、カテゴリごとにまとめられて並んでいることがあります。
# そのとき、データセットの先頭から 18 個目のところで訓練データとテストデータに分割したとすると、訓練データには目標値が 3 のデータが 1 つも含まれないこととなります。
#
# そこで、ランダムにデータセットを分割する方法が採用されています。
# `random_state` という引数に毎回同じ整数を与えることで、実行のたびに結果が変わることを防いでいます。
#
# それでは、分割後の訓練データを用いてモデルの訓練、精度の検証を行いましょう。

# + [markdown] colab_type="text" id="l71nhJfL1eWP"
# ### Step 2 ~ 4：モデル・目的関数・最適化手法を決める
#
# scikit-learn で重回帰分析を行う場合は、`LinearRegression` クラスを使用します。
# `sklearn.linear_model` 以下にある `LinearRegression` クラスを読み込んで、インスタンスを作成しましょう。

# + colab={} colab_type="code" id="QLxsQ11I3OG-"
from sklearn.linear_model import LinearRegression

# モデルの定義
reg_model = LinearRegression()

# + [markdown] colab_type="text" id="fWcEwctR36Dj"
# 上記のコードは、前述の 2 〜 4 までのステップを行います。
# `LinearRegression` は最小二乗法を行うクラスで、目的関数や最適化手法もあらかじめ内部で用意されたものが使用されます。
# 詳しくはこちらのドキュメントを参照してください。
# 参考：[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
# -

# ### Step 5：モデルの訓練
#
# 次にモデルの訓練を行います。
# scikit-learn に用意されている手法の多くは、共通して `fit()` というメソッドを持っており、
# 再利用可能なコードが書きやすくなっています。
#
# `reg_model` を用いて訓練を実行するには、`fit()` の引数に入力値 `x` と目標値 `t` を与えます。

# + colab={"base_uri": "https://localhost:8080/", "height": 53} colab_type="code" id="DJbiWfeg3V1L" outputId="31392ce6-eb54-4705-c678-b45123482b55"
# モデルの訓練
reg_model.fit(x_train, t_train)

# + [markdown] colab_type="text" id="4h-Isref4kCB"
# モデルの訓練が完了しました。
# 求まったパラメータの値を確認してみましょう。
# 重回帰分析では、重み `w` とバイアス `b` の２つがパラメータでした。
# 求まった重み `w` の値は `model.coef_` に、バイアス `b` の値は `model.intercept_` に格納されています。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Q_-9poc-3cft" outputId="d52f84f2-d786-4622-f35e-c25baf9f7e77"
# 訓練後のパラメータ w
reg_model.coef_

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="2rozWBiq1eWt" outputId="f02bf67f-b3ca-40cc-af7e-620030f02ffa"
# 訓練後のバイアス b
reg_model.intercept_

# + [markdown] colab_type="text" id="hQXB57Gr4xd_"
# モデルの訓練が完了したら、精度の検証を行います。
# `LinearRegression` クラスは `score()` メソッドを提供しており、入力値と目標値を与えると訓練済みのモデルを用いて計算した**決定係数 (coefficient of determination)** という指標を返します。
#
# これは、使用するデータセットのサンプルサイズを $N$、$n$ 個目の入力値に対する予測値を $y_{n}$、目標値を $t_n$、そしてそのデータセット内の全ての目標値の平均値を $\bar{t}$ としたとき、
#
# $$
# R^{2} = 1 - \dfrac{\sum_{n=1}^{N}\left( t_{n} - y_{n} \right)^{2}}{\sum_{n=1}^{N}\left( t_{n} - \bar{t} \right)^{2}}
# $$
#
# で表される指標です。
#
# 決定係数の最大値は 1 であり、値が大きいほどモデルが与えられたデータに当てはまっていることを表します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6uNksu9X3pMF" outputId="74c0bada-9f74-4df5-d260-17e8ea1c5a99"
# 精度の検証
reg_model.score(x_train, t_train)
# -

# 訓練済みモデルを用いて訓練用データセットで計算した決定係数は、およそ 0.765 でした。

# + [markdown] colab_type="text" id="ndGdp3kB4ZPZ"
# ### 新しい入力値に対する予測の計算（推論）
#
# 訓練が終わったモデルに、新たな入力値を与えて、予測値を計算させるには、`predict()` メソッドを用います。
# 訓練済みのモデルを使ったこのような計算は、**推論 (inference)** と呼ばれることがあります。
# 今回は、訓練済みモデル `reg_model` を用いて、テスト用データセットからサンプルを 1 つ取り出し、推論を行ってみましょう。
# このとき、`predict()` メソッドに与える入力値の ndarray の形が `(サンプルサイズ, 各サンプルの次元数)` となっている必要があることに気をつけてください。

# + colab={} colab_type="code" id="e84j6bbh51Wg"
reg_model.predict(x_test[:1])
# -

# この入力に対する目標値を見てみましょう。

t_test[0]

# 22.6 という目標値に対して、およそ 24.94 という予測値が返ってきました。

# ### テスト用データセットを用いた評価
#
# それでは、訓練済みモデルの性能を、テスト用データセットを使って決定係数を計算することで評価してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="zSiL_fmE56W4" outputId="dee773a5-619d-4153-d642-2ebf5010dd0e"
reg_model.score(x_test, t_test)
# -

# 訓練用データセットを用いて算出した値（およそ 0.765）よりも、低い値がでてしまいました。
#
# 教師あり学習の目的は、訓練時には見たことがない新しいデータ、ここではテスト用データセットに含まれているデータに対しても、高い性能を発揮するように、モデルを訓練することです。
# 逆に、訓練時に用いたデータに対してはよく当てはまっていても、訓練時に用いなかったデータに対しては予測値と目標値の差異が大きくなってしまう現象を、**過学習 (overfitting)** と言います。
#
# 過学習を防ぐために、色々な方法が研究されています。
# ここでは、データに前処理を行い、テスト用データセットを用いて計算した決定係数を改善します。

# ## 各ステップの改善

# + [markdown] colab_type="text" id="dwpv2FxdFSYi"
# ### Step 1 の改善：前処理
#
# **前処理 (preprocessing)** とは、欠損値の補完、外れ値の除去、特徴量選択、正規化などの処理を訓練を開始する前にデータセットに対して行うことです。
#
# 手法やデータに合わせた前処理が必要となるため、適切な前処理を行うためには手法そのものについて理解している必要があるだけでなく、使用するデータの特性についてもよく調べておく必要があります。
#
# 今回のデータは、入力値の値の範囲が CRIM, ZN, INDUS, ... といった指標ごとに大きく異なっています。
# そこで、各入力変数ごとに平均が 0、分散が 1 となるように値をスケーリングする**標準化 (standardization)** をおこなってみましょう。
#
# scikit-learn では `sklearn.preprocessing` というモジュール以下に `StandardScaler` というクラスが定義されています。
# 今回は、これを用いてデータセットに対し標準化を適用します。
# それでは、`StandardScaler` クラスを読み込み、インスタンスを作成します。

# + colab={} colab_type="code" id="GLQNMom0CZXA"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# + [markdown] colab_type="text" id="X_CsbDW_1eX4"
# 標準化を行うためには、データセットの各入力変数ごとの平均と分散の値を計算する必要があります。
# この計算は、`scaler` オブジェクトがもつ `fit()` メソッドを用いて行います。
# 引数には、平均・分散を計算したい入力値の ndarray を渡します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="YROpwqRhCZca" outputId="166fe5d5-6e91-46bb-b32c-8371e2dc25bb"
scaler.fit(x_train)

# + [markdown] colab_type="text" id="LmJsOAbP1eX7"
# すべてのサンプルではなく、訓練用データセットのみを用いてこれらの値を算出します。
# 先ほどの `fit()` の実行の結果、算出された平均値が `mean_` 属性に、分散が `var_` 属性に格納されているので、確認してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" id="nbWSTWHB1eX8" outputId="31ed1b63-e065-4012-b6bd-36616b9b59a1"
# 平均
scaler.mean_

# + colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" id="oAXIw1831eX9" outputId="da1ec61a-10ac-4172-ebf9-ed2165dcd78c"
# 分散
scaler.var_

# + [markdown] colab_type="text" id="zYZml5H31eX-"
# これらの平均・分散の値を使って、データセットに含まれる値に標準化を施すには、`transform()` メソッドを使用します。

# + colab={} colab_type="code" id="1sz88ORaLw3_"
x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)
# -

# それでは、標準化を行ったデータを使って、同じモデルを訓練してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="kdGJ0lNnLxJY" outputId="a387ac05-7abb-491c-c1fa-637818ba374c"
reg_model = LinearRegression()

# モデルの訓練
reg_model.fit(x_train_scaled, t_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="70X7rqb3LxW3" outputId="0e99fde5-61de-404f-8bf6-62e78b739eed"
# 精度の検証（訓練データ）
reg_model.score(x_train_scaled, t_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="rd1uZ0_Y1eYE" outputId="a57104e4-4b5e-4019-88de-20f22cc71512"
# 精度の検証（テストデータ）
reg_model.score(x_test_scaled, t_test)
# -

# 結果は変わりませんでした。
#
# べき変換をする別の前処理を適用し、再度同じモデルの訓練を行ってみましょう。

# +
from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model = LinearRegression()
reg_model.fit(x_train_scaled, t_train)
# -

# 訓練データでの決定係数
reg_model.score(x_train_scaled, t_train)

# テストデータでの決定係数
reg_model.score(x_test_scaled, t_test)

# + [markdown] colab_type="text" id="4rkchSc_1eYG"
# 結果が改善しました。

# + [markdown] colab_type="text" id="iNERbB_21eYG"
# #### パイプライン化
#
# 前処理用の `scaler` と 重回帰分析を行う `reg_model` は、両方 `fit()` メソッドを持っていました。
# scikit-learn には、パイプラインと呼ばれる一連の処理を統合できる機能があります。
# これを用いて、これらの処理をまとめてみましょう。

# + colab={} colab_type="code" id="qudHL8J51eYG"
from sklearn.pipeline import Pipeline

# パイプラインの作成 (scaler -> svr)
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('reg', LinearRegression())
])

# + colab={"base_uri": "https://localhost:8080/", "height": 109} colab_type="code" id="CmpM51eh1eYK" outputId="7493f796-dead-45ac-dfd1-e11fefb52d70"
# scaler および reg を順番に使用
pipeline.fit(x_train, t_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="jz8N8fkf1eYM" outputId="43f1725a-2558-443b-83e1-38638e88ae15"
# 訓練用データセットを用いた決定係数の算出
pipeline.score(x_train, t_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="ebbWBAC31eYN" outputId="76bffbed-c486-4f31-a403-fe141bae673e"
# テスト用データセットを用いた決定係数の算出
linear_result = pipeline.score(x_test, t_test)

linear_result

# + [markdown] colab_type="text" id="NSPXVQxK1eYO"
# パイプライン化を行うことで、`x_train_scaled` のような中間変数を作成することなく、同じ処理が行えるようになりました。
# これによってコード量が減らせるだけでなく、評価を行う前にテスト用データセットに対しても訓練用データセットに対して行ったのと同様の前処理を行うことを忘れてしまうといったミスを防ぐことができます。
