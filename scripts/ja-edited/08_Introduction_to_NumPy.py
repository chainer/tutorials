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

# + [markdown] colab_type="text" id="3-ewqmV8hszX"
# # NumPy 入門
#
# 本章では、Python で数値計算を高速に行うためのライブラリ（[注釈1](#note1)）である NumPy の使い方を学びます。
# 本章の目標は、[単回帰分析と重回帰分析](https://tutorials.chainer.org/ja/07_Regression_Analysis.html)の章で学んだ重回帰分析を行うアルゴリズムを**NumPy を用いて実装すること**です。
#
# NumPy による**多次元配列（multidimensional array）**の扱い方を知ることは、他の様々なライブラリを利用する際に役立ちます。
# 例えば、様々な機械学習手法を統一的なインターフェースで利用できる **scikit-learn** や、ニューラルネットワークの記述・学習を行うためのフレームワークである **Chainer** は、NumPy に慣れておくことでとても使いやすくなります。
#
# それでは、まず NumPy の基礎的な使用方法を説明します。

# + [markdown] colab_type="text" id="iPmvXIQThszZ"
# ## NumPy を使う準備
#
# NumPy は Google Colaboratory（以下 Colab）上のノートブックにはデフォルトでインストールされているため、ここではインストールの方法は説明しません。自分のコンピュータに NumPy をインストールしたい場合は、こちらを参照してください。：[Installing packages](https://scipy.org/install.html)
#
# Colab 上ではインストール作業は必要ないものの、ノートブックを開いた時点ではまだ `numpy` モジュールが読み込まれていません。
# ライブラリの機能を利用するには、そのライブラリが提供するモジュールを読み込む必要があります。
#
# 例えば `A` というモジュールを読み込みたいとき、一番シンプルな記述方法は `import A` です。
# ただ、もし `A` というモジュール名が長い場合は、`import A as B` のようにして別名を付けることができます。
# `as` を使って別名が与えられると、以降そのモジュールはその別名を用いて利用することができます。
# `import A as B` と書くと、`A` というモジュールは `B` という名前で利用することができます。
# これは Python の機能なので NumPy 以外のモジュールを読み込みたい場合にも使用可能です。
#
# 慣習的に、`numpy` にはしばしば `np` という別名が与えられます。
# コード中で頻繁に使用するモジュールには、短い別名をつけて定義することがよく行われます。
#
# それでは、`numpy` を `np` という名前で `import` してみましょう。

# + colab={} colab_type="code" id="YGm0nN_vuWfY"
import numpy as np

# + [markdown] colab_type="text" id="SB5_Yx1qXAcN"
# ## 多次元配列を定義する
#
# ベクトル・行列・テンソルなどは、プログラミング上は多次元配列により表現でき、NumPy では ndarray というクラスで多次元配列を表現します（[注釈2](#note2)）。早速、これを用いてベクトルを定義してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="HTartJBpvinm" outputId="0731ea56-3ac2-4ec5-c073-69a029d6d2c3"
# ベクトルの定義
a = np.array([1, 2, 3])

a

# + [markdown] colab_type="text" id="LCbcrkX6hszo"
# このように、Python リスト `[1, 2, 3]` を `np.array()` に渡すことで、$[1, 2, 3]$ というベクトルを表す ndarray オブジェクトを作ることができます。
# ndarray オブジェクトは `shape` という**属性 （attribute）** を持っており、その多次元配列の**形 （shape）** が保存されています。
# 上で定義した `a` という ndarray オブジェクトの形を調べてみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="O2GmnrGVhszq" outputId="b3cf41ce-66f9-434a-9082-22af7ed8ff9c"
a.shape

# + [markdown] colab_type="text" id="YlU3JFZVhszw"
# `(3,)` という要素数が 1 の Python のタプルが表示されています。
# ndarray の形は、要素が整数のタプルで表され、要素数はその多次元配列の**次元数 （dimensionality, number of dimensions）** を表します。
# 形は、その多次元配列の各次元の大きさを順に並べた整数のタプルになっています。
#
# 次元数は、ndarray の `ndim` という属性に保存されています。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TClptSKihszx" outputId="1a22e147-f5cb-4cca-d586-351891bef7dc"
a.ndim

# + [markdown] colab_type="text" id="ALEfYmAuhsz1"
# これは、`len(a.shape)` と同じ値になります。
# 今、`a` という ndarray は 1 次元配列なので、`a.shape` は要素数が 1 のタプルで、`ndim` の値は 1 でした（[注釈3](#note3)）。

# + [markdown] colab_type="text" id="oQcB5yZvhsz2"
# では次に、$3 \times 3$ 行列を定義してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="u3MkTwo0jM1y" outputId="daed274d-9d7f-43b4-9b3b-3f729ce821ad"
# 行列の定義
b = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)

b

# + [markdown] colab_type="text" id="NEZor0zths0B"
# 形と次元数を調べます。

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="VdoPp_H3hs0C" outputId="d326c3de-28b0-40ba-d594-ff6befef3ceb"
print('Shape:', b.shape)
print('Rank:', b.ndim)

# + [markdown] colab_type="text" id="GRcKmhhRhs0G"
# ここで、`size` という属性も見てみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="nL7kUxyThs0H" outputId="a5a99a98-b7a6-4ff4-f71e-841259bf2801"
b.size

# + [markdown] colab_type="text" id="6qHL5knGhs0L"
# これは、`b` という ndarray が持つ要素の数を表しています。
# `b` は $3 \times 3$ 行列なので、要素数は 9 です。
# **「形」「次元数」「サイズ」という言葉がそれぞれ意味するものの違いを確認してください。**
#
# NumPy の ndarray の作成方法には、`np.array()` を用いて Python のリストから多次元配列を作る方法以外にも、色々な方法があります。
# 以下に代表的な例をいくつか紹介します。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="-E2Rtwq9hs0N" outputId="8a89d4fb-508c-4328-8001-6837b331eed2"
# 形を指定して、要素が全て 0 で埋められた ndarray を作る
a = np.zeros((3, 3))

a

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="870IpdMyhs0R" outputId="b45b6bf4-22b8-495f-c15f-6e6467ef6fb9"
# 形を指定して、要素が全て 1 で埋められた ndarray を作る
b = np.ones((2, 3))

b

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="xndboQKyhs0W" outputId="1371ef3d-043c-48dc-de8c-6d8d05f9cfdf"
# 形と値を指定して、要素が指定した値で埋められた ndarray を作る
c = np.full((3, 2), 9)

c

# + colab={"base_uri": "https://localhost:8080/", "height": 106} colab_type="code" id="WA5IUfrmhs0b" outputId="b3162424-a0a9-4110-dc3b-cb291a90ac91"
# 指定された大きさの単位行列を表す ndarray を作る
d = np.eye(5)

d

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="YLH4LMVThs0h" outputId="97b95b3c-ace3-4caf-ec13-d4e01d5d3c7d"
# 形を指定して、 0 ~ 1 の間の乱数で要素を埋めた ndarray を作る
e = np.random.random((4, 5))

e

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="JrlPdcOths0r" outputId="016d37ff-60d6-4aa1-cdb2-ad58154e10fe"
# 3 から始まり 10 になるまで 1 ずつ増加する数列を作る（10 は含まない）
f = np.arange(3, 10, 1)

f

# + [markdown] colab_type="text" id="1rPZMxVIhs0w"
# ## 多次元配列の要素を選択する
#
# 前節では NumPy を使って多次元配列を定義するいくつかの方法を紹介しました。
# 本節では、作成した ndarray のうちの特定の要素を選択して、値を取り出す方法を紹介します。
# 最もよく行われる方法は `[]` を使った**添字表記 （subscription）** による要素の選択です。

# + [markdown] colab_type="text" id="yjqfy8AVhs0x"
# ### 整数による要素の選択
#
# 例えば、上で作成した `e` という $4 \times 5$ 行列を表す多次元配列から、1 行 2 列目の値を取り出すには、以下のようにします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="4wXjOOjyhs0y" outputId="6581d4a6-dc3c-46a5-cc00-c91923b2e7a4"
val = e[0, 1]

val

# + [markdown] colab_type="text" id="FPYVW5rqhs02"
# 「1 行 2 列目」を指定するのに、インデックスは `[0, 1]` でした。
# これは、NumPy の ndarray の要素は Python リストと同じく、添字が 0 から始まる**ゼロベースインデックス （zero-based index）** が採用されているためです。
# つまり、この行列の i 行 j 列目の値は、`[i - 1, j - 1]` で取り出すことができます。

# + [markdown] colab_type="text" id="qWnR7Eurhs03"
# ### スライスによる要素の選択
#
# NumPy の ndarray に対しても、Python のリストと同様に**スライス表記 （slicing）** を用いて選択したい要素を範囲指定することができます。
# ndarray はさらに、カンマ区切りで複数の次元に対するスライスを指定できます。

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="Q0nqhCpfhs04" outputId="065d1646-ba1a-4b0c-92ef-36ea788ab8aa"
# 4 x 5 行列 e の真ん中の 2 x 3 = 6 個の値を取り出す
center = e[1:3, 1:4]

center

# + [markdown] colab_type="text" id="1FUL2bOkhs09"
# 前節最後にある `e` の出力を見返すと、ちょうど真ん中の部分の $2 \times 3$ 個の数字が取り出せていることが分かります。
# ここで、`e` の中から `[1, 1]` の要素を起点として 2 行 3 列を取り出して作られた `center` の形を、`e` の形と比較してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="g57LOQJbhs0_" outputId="5ad96c57-27ec-4a5e-99e0-f40747463619"
print('Shape of e:', e.shape)
print('Shape of center:', center.shape)

# + [markdown] colab_type="text" id="5tbJR05ehs1F"
# また、インデックスを指定したり、スライスを用いて取り出した ndarray の一部に対し、値を代入することもできます。

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="pnJGJ0sOhs1G" outputId="5b3be8f5-918e-4aaf-c304-4b08e52a0754"
# 先程の真ん中の 6 個の値を 0 にする
e[1:3, 1:4] = 0

e

# + [markdown] colab_type="text" id="7n7aJGJahs1J"
# ### 整数配列による要素の選択
#
# ndarray の `[]` には、整数やスライスの他に、整数配列を渡すこともできます。
# 整数配列とは、ここでは整数を要素とする Python リストまたは ndarray のことを指しています。
#
# 具体例を示します。
# まず、$3 \times 3$ 行列を表す `a` という ndarray を定義します。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="XxgfjK2-hs1L" outputId="83abf535-d424-4c6c-d37a-4f0b0124a25f"
a = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)

a

# + [markdown] colab_type="text" id="qjX-7iNths1R"
# この ndarray から、
#
# 1. 1 行 2 列目：`a[0, 1]`
# 2. 3 行 2 列目：`a[2, 1]`
# 3. 2 行 1 列目：`a[1, 0]`
#
# の 3 つの要素を選択して並べ、形が `(3,)` であるような ndarray を作りたいとします。
#
# これは、以下のように、順に対象の要素を指定して並べて新しい ndarray にすることでももちろん実現できます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="UvEgg_eHhs1S" outputId="3f27d554-9d09-4327-8138-4d2b5c085b16"
np.array([a[0, 1], a[2, 1], a[1, 0]])

# + [markdown] colab_type="text" id="r2aNYWNFhs1V"
# しかし、同じことが**選択したい行、選択したい列を、順にそれぞれリストとして与える**ことでも行えます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="P1scHQiRhs1W" outputId="e0c1fe5d-4f63-4ba3-e3e9-8540b86f3b3e"
a[[0, 2, 1], [1, 1, 0]]

# + [markdown] colab_type="text" id="0sHruq-ohs1Z"
# **選択したい 3 つの値がどの行にあるか**だけに着目すると、それぞれ 1 行目、3 行目、2 行目にある要素です。  
# ゼロベースインデックスでは、それぞれ 0, 2, 1 行目です。  
# これが `a` の `[]` に与えられた 1 つ目のリスト `[0, 2, 1]` の意味です。  
#
# 同様に、**列に着目**すると、ゼロベースインデックスでそれぞれ 1, 1, 0 列目の要素です。  
# これが `a` の `[]` に与えられた 2 つ目のリスト `[1, 1, 0]` の意味です。

# + [markdown] colab_type="text" id="HDdtMWR4hs1a"
# ## ndarray のデータ型
#
# 1 つの ndarray の要素は、全て同じ型を持ちます。
# NumPy では様々なデータ型を使うことができますが、ここでは一部だけを紹介します。
# NumPy は Python リストを渡して ndarray を作る際などには、その値からデータ型を推測します。
# ndarray のデータ型は、`dtype` という属性に保存されています。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="eU6frncOhs1d" outputId="3f764b12-bb3c-4a65-89ad-159d980a2c34"
# 整数（Python の int 型）の要素をもつリストを与えた場合
x = np.array([1, 2, 3])

x.dtype

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="3C4OQKq9hs1k" outputId="f46446f4-a54b-4278-846f-86b82e8c32d1"
# 浮動小数点数（Python の float 型）の要素をもつリストを与えた場合
x = np.array([1., 2., 3.])

x.dtype

# + [markdown] colab_type="text" id="HTg_bqiuhs1o"
# 以上のように、**Python の int 型は自動的に NumPy の int64 型**になりました。
# また、**Python の float 型は自動的に NumPy の float64 型**になりました。
# Python の int 型は NumPy の int_ 型に対応づけられており、Python の float 型は NumPy の float_ 型に対応づけられています。
# この int_ 型はプラットフォームによって int64 型と同じ場合と int32 型と同じ場合があります。
# float_ 型についても同様で、プラットフォームによって float64 型と同じ場合と float32 型と同じ場合があります。
#
# 特定の型を指定して ndarray を作成するには、以下のようにします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="3UZiigfFhs1p" outputId="c422ade7-f6dc-412f-c40f-9ee18c47685b"
x = np.array([1, 2, 3], dtype=np.float32)

x.dtype

# + [markdown] colab_type="text" id="aWMTSehbhs1s"
# このように、`dtype` という引数に NumPy の dtype オブジェクトを渡します。
# これは 32 ビット浮動小数点数型を指定する例です。
# 同じことが、文字列で指定することによっても行えます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="jg3-zxU2yEpd" outputId="f59a3c33-7b99-42a3-8079-46d7bdd9b28e"
x = np.array([1, 2, 3], dtype='float32')

x.dtype

# + [markdown] colab_type="text" id="UZ2hdYZ_yEph"
# これはさらに、以下のように短く書くこともできます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="tnn0EQQmyEpk" outputId="84a57ee4-0b87-40c2-ccb6-c1672b71f27a"
x = np.array([1, 2, 3], dtype='f')

x.dtype

# + [markdown] colab_type="text" id="wa5IPghnyEps"
# 一度あるデータ型で定義した配列のデータ型を別のものに変更するには、`astype` を用いて変換を行います。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="fnqEBxUMhs1t" outputId="1191035e-7a37-4a43-fbe6-70f59fe0fcc8"
x = x.astype(np.float64)

x.dtype

# + [markdown] colab_type="text" id="gQtt_qLLjeLQ"
# ## 多次元配列を用いた計算
#
# ndarray を使って行列やベクトルを定義して、それらを用いていくつかの計算を行ってみましょう。
#
# ndarray として定義されたベクトルや行列同士の**要素ごとの加減乗除**は、Python の数値同士の四則演算に用いられる `+`、`-`、`*`、`/` という記号を使って行えます。
#
# それでは、同じ形の行列を 2 つ定義し、それらの**要素ごとの**加減乗除を実行してみましょう。

# + colab={} colab_type="code" id="_hpNMPZpw24s"
# 同じ形 (3 x 3) の行列を 2 つ定義する
a = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="O5zs75ooyEp2" outputId="1965d192-50d8-4cb0-9f3c-ac294f27db4a"
# 足し算
c = a + b

c

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="c2FOyv2YyEp5" outputId="fd28f151-0281-4e4f-a787-11db1d5ceeb0"
# 引き算
c = a - b

c

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="Pyiqo_ZYyEp8" outputId="436b891f-7d3b-4d30-afd2-7bca3f929955"
# 掛け算
c = a * b

c

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="Ut6N5zcTyEqA" outputId="6eda2f36-e116-47d0-a45c-afc344f4c968"
# 割り算
c = a / b

c

# + [markdown] colab_type="text" id="HFZIVmCwhs13"
# NumPy では、与えられた多次元配列に対して要素ごとに計算を行う関数が色々と用意されています。
# 以下にいくつかの例を示します。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="BUaEHDrcyEqL" outputId="57186cb3-188c-4b92-b354-8839496fa1de"
# 要素ごとに平方根を計算する
c = np.sqrt(b)

c

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="X9XlfvFths14" outputId="94d8138e-36b9-4b7f-fe90-fc8bce1ee4a4"
# 要素ごとに値を n 乗する
n = 2
c = np.power(b, n)

c

# + [markdown] colab_type="text" id="XC-OB1s_hs19"
# 要素ごとに値を n 乗する計算は、以下のようにしても書くことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="7ekgNzVlhs2F" outputId="c1b1deb1-4f7c-45e3-fd02-a0319d00219a"
c ** n

# + [markdown] colab_type="text" id="9exqunixhs2K"
# はじめに紹介した四則演算は、**同じ大きさの** 2 つの行列同士で行っていました。
# ここで、$3 \times 3$ 行列 `a` と 3 次元ベクトル `b` という大きさのことなる配列を定義して、それらを足してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="geIUaMashs2L" outputId="0cb165b3-98fc-477d-fc97-1c4f23e4c5e2"
a = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

b = np.array([1, 2, 3])

c = a + b

c

# + [markdown] colab_type="text" id="9dns4SENhs2T"
# 形が同じ行列同士の場合と同様に計算することができました。
#
# これは NumPy が自動的に**ブロードキャスト（broadcast）**と呼ばれる操作を行っているためです。
# これについて次節で説明します。

# + [markdown] colab_type="text" id="zD5qR1Dvhs2V"
# ## ブロードキャスト
#
# 行列同士の要素ごとの四則演算は、通常は行列の形が同じでなければ定義できません。
# しかし、前節の最後では $3 \times 3$ 行列に 3 次元ベクトルを足す計算が実行できました。
#
# これが要素ごとの計算と同じように実行できる理由は、NumPy が自動的に 3 次元ベクトル `b` を 3 つ並べてできる $3 \times 3$ 行列を想定し、`a` と同じ形に揃える操作を暗黙に行っているからです。
# この操作を、**ブロードキャスト**と呼びます。
#
# 算術演算を異なる形の配列同士で行う場合、NumPy は自動的に小さい方の配列を**ブロードキャスト**し、大きい方の配列と形を合わせます。
# ただし、この自動的に行われるブロードキャストでは、行いたい算術演算が、大きい方の配列の一部に対して**繰り返し行われる**ことで実現されるため、実際に小さい方の配列のデータをコピーして大きい配列をメモリ上に作成することは可能な限り避けられます。
# また、この繰り返しの計算は NumPy の内部の C 言語によって実装されたループで行われるため、高速です。
#
# よりシンプルな例で考えてみましょう。
# 以下のような配列 `a` があり、この全ての要素を 2 倍にしたいとします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="65YCT4dzhs2W" outputId="df386959-2091-477f-9669-d9f4d72a8f80"
a = np.array([1, 2, 3])

a

# + [markdown] colab_type="text" id="Zp1-yKjXhs2a"
# このとき、一つの方法は以下のように同じ形で要素が全て 2 である別の配列を定義し、これと要素ごとの積を計算するやり方です。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="-HXN6pyVhs2b" outputId="ffcd209d-a7c2-412e-b79c-fe6955675ff3"
b = np.array([2, 2, 2])

c = a * b

c

# + [markdown] colab_type="text" id="Y7XJ7T6Mhs2g"
# しかし、スカラの 2 をただ `a` に掛けるだけでも同じ結果が得られます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="XUC3S8jqhs2h" outputId="a27f7d90-c84a-4339-c152-a185dfc60a42"
c = a * 2

c

# + [markdown] colab_type="text" id="SLje0bp-hs2k"
# `* 2` という計算が、`c` の 3 つの要素の**どの要素に対する計算なのか**が明示されていないため、NumPy はこれを**全ての要素に対して行うという意味**だと解釈して、スカラの 2 を `a` の要素数 3 だけ引き伸ばしてから掛けてくれます。
#
# **形の異なる配列同士の計算がブロードキャストによって可能になるためにはルールがあります。**
#
# それは、**「2 つの配列の各次元が同じ大きさになっているか、どちらかが 1 であること」**です。
# このルールを満たさない場合、NumPy は "ValueError: operands could not be broadcast together with shapes (1 つ目の配列の形) (2 つ目の配列の形)" というエラーを出します。
#
# ブロードキャストされた配列の各次元のサイズ（[注釈4](#note4)）は、入力された配列のその次元のサイズの中で最大の値と同じになっています。
# 入力された配列は、各次元のサイズが入力のうち大きい方のサイズと同じになるようブロードキャストされ、その拡張されたサイズで計算されます。
#
# もう少し具体例を見てみましょう。
# 以下のような 2 つの配列 `a` と `b` を定義し、足します。

# + colab={"base_uri": "https://localhost:8080/", "height": 444} colab_type="code" id="bD4dr8ilhs2l" outputId="c2a409b2-2d67-4ad4-85d1-ce8ed42a03c3"
# 0 ~ 9 の範囲の値をランダムに用いて埋められた (2, 1, 3) と (3, 1) という大きさの配列を作る
a = np.random.randint(0, 10, (2, 1, 3))
b = np.random.randint(0, 10, (3, 1))

print('a:\n', a)
print('\na.shape:', a.shape)
print('\nb:\n', b)
print('\nb.shape:', b.shape)

# 加算
c = a + b

print('\na + b:\n', c)
print('\n(a + b).shape:', c.shape)

# + [markdown] colab_type="text" id="o90A_pvhhs2u"
# `a` の形は `(2, 1, 3)` で、`b` の形は `(3, 1)` でした。
# この 2 つの配列の**末尾次元 (trailing dimension)**（[注釈5](#note5)） はそれぞれ 3 と 1 なので、ルールにあった「次元が同じサイズであるか、どちらかが 1 であること」を満たしています。
#
# 次に、各配列の第 2 次元に注目してみましょう。
# それぞれ 1 と 3 です。
# これもルールを満たしています。
#
# ここで、`a` は 3 次元配列ですが、`b` は 2 次元配列です。
# つまり、次元数が異なっています。
# このような場合は、`b` は**一番上の次元にサイズが 1 の次元が追加された形** `(1, 3, 1)` として扱われます。
# そして 2 つの配列の各次元ごとのサイズの最大値をとった形 `(2, 3, 3)` にブロードキャストされ、足し算が行われます。
#
# このように、もし 2 つの配列のランクが異なる場合は、次元数が小さい方の配列が大きい方と同じ次元数になるまでその形の先頭に新たな次元が追加されます。
# サイズが 1 の次元がいくつ追加されても、要素の数は変わらないことに注意してください。
# 要素数（`size` 属性で取得できる値）は、各次元のサイズの掛け算になるので、1 を何度かけても値は変わらないことから、これが成り立つことが分かります。
#
# NumPy がブロードキャストのために自動的に行う新しい次元の挿入は、`[]` を使った以下の表な表記を用いることで**手動で行うこともできます。**

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="6TwzCz9khs2v" outputId="b9390767-8ba0-4567-a7a7-dd5999cf4673"
print('Original shape:', b.shape)

b_expanded = b[np.newaxis, :, :]

print('Added new axis to the top:', b_expanded.shape)

b_expanded2 = b[:, np.newaxis, :]

print('Added new axis to the middle:', b_expanded2.shape)

# + [markdown] colab_type="text" id="g3CSUaHAhs20"
# `np.newaxis` が指定された位置に、新しい次元が挿入されます。
# 配列が持つ数値の数は変わっていません。
# そのため、挿入された次元のサイズは必ず 1 になります。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="eK7NmjCWhs21" outputId="cacfe9cc-76ec-4aec-ef10-d93662a75d3c"
b

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="wappsylzhs2_" outputId="30ed68c3-b11a-48d3-98bc-04fc0ea4fbdf"
b_expanded

# + colab={"base_uri": "https://localhost:8080/", "height": 106} colab_type="code" id="l-IHRhU1hs3E" outputId="1b1c6f57-b8f6-421b-bb18-b254d3b4b355"
b_expanded2

# + [markdown] colab_type="text" id="Mv8XZb4Bhs3I"
# NumPy のブロードキャストは慣れるまで直感に反するように感じる場合があるかもしれません。
# しかし、使いこなすと同じ計算が Python のループを使って行うよりも高速に行えるため、ブロードキャストを理解することは非常に重要です。
# 一つ具体例を見てみます。
#
# $5 \times 5$ 行列 `a` に、3 次元ベクトル `b` を足します。
# まず、`a`、`b` および結果を格納する配列 `c` を定義します。

# + colab={} colab_type="code" id="KW3-2Gz3hs3J"
a = np.array([
    [0, 1, 2, 1, 0],
    [3, 4, 5, 4, 3],
    [6, 7, 8, 7, 6],
    [3, 4, 5, 4, 4],
    [0, 1, 2, 1, 0]
])

b = np.array([1, 2, 3, 4, 5])

# 結果を格納する配列を先に作る
c = np.empty((5, 5))

# + [markdown] colab_type="text" id="CaPTRYq1hs3L"
# `%%timeit` という Jupyter Notebook で使用できるそのセルの実行時間を計測するためのマジックを使って、`a` の各行（1 次元目）に `b` の値を足していく計算を Python のループを使って 1 行ずつ処理していくコードの実行時間を測ってみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="aTJ5owYUhs3M" outputId="91ad0294-e6ba-4b39-d536-a5563b1bd612"
# %%timeit
for i in range(a.shape[0]):
    c[i, :] = a[i, :] + b

# + colab={"base_uri": "https://localhost:8080/", "height": 106} colab_type="code" id="-6z4EGzXhs3Q" outputId="ea6f591f-4841-45b6-f460-7bd7f9a032c7"
c

# + [markdown] colab_type="text" id="3KXentjEhs3T"
# 次に、NumPy のブロードキャストを活用した方法で同じ計算を行ってみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="pCgJcwt7hs3T" outputId="5af58db9-d332-4d80-ee1c-938df607a7e6"
# %%timeit
c = a + b

# + colab={"base_uri": "https://localhost:8080/", "height": 106} colab_type="code" id="EAetPXrphs3Z" outputId="8d13e5ef-e68a-4bcd-d507-0086653a78ff"
c

# + [markdown] colab_type="text" id="USEUvvcChs3c"
# 計算結果は当然同じになります。
# しかし、実行時間が数倍短くなっています。
#
# このように、ブロードキャストを理解して活用することで、記述が簡単になるだけでなく、実行速度という点においても有利になります。

# + [markdown] colab_type="text" id="8Fd_G9JckA0Z"
# ## 行列積
#
# 行列の要素ごとの積は `*` を用いて計算できました。
# 一方、通常の行列同士の積（行列積）の計算は、`*` ではなく、別の方法で行います。
# 方法は 2 種類あります。
#
# 1つは、`np.dot()` 関数を用いる方法です。
# `np.dot()` は 2 つの引数をとり、それらの行列積を計算して返す関数です。
# 今、`A` という行列と `B` という行列があり、行列積 `AB` を計算したいとします。
# これは `np.dot(A, B)` と書くことで計算できます。
# もし `BA` を計算したい場合は、`np.dot(B, A)` と書きます。
#
# もう 1 つは、ndarray オブジェクトが持つ `dot()` メソッドを使う方法です。
# これを用いると、同じ計算が `A.dot(B)` と書くことによって行えます。

# + colab={} colab_type="code" id="_rMylHNEhpFs"
# 行列 A の定義
A = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

# 行列 B の定義
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# + [markdown] colab_type="text" id="P-mVIroIhs3j"
# 実際にこの $3 \times 3$ の 2 つの行列の行列積を計算してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="Go9DfbmXhs3l" outputId="6b11a289-b91f-4944-fb63-f41f61feb496"
# 行列積の計算 (1)
C = np.dot(A, B)

C

# + [markdown] colab_type="text" id="FNU92PFPhs3o"
# 同じ計算をもう一つの記述方法で行ってみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="-YmjkeC8hs3p" outputId="a6b3774a-f02f-4029-d721-f15140be7e05"
C = A.dot(B)

C

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="E6UqcewwoFOv" outputId="be98d878-fdc0-4ea3-b152-cc093bd3628a"
# データ型の確認（整数値）
a.dtype

# + [markdown] colab_type="text" id="4SH7NONRpfO1"
# ## 基本的な統計量の求め方
#
# 本節では、多次元配列に含まれる値の平均・分散・標準偏差・最大値・最小値といった統計値を計算する方法を紹介します。
# $8 \times 10$ の行列を作成し、この中に含まれる値全体に渡るこれらの統計値を計算してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 159} colab_type="code" id="QtRVfBYahs3w" outputId="abeeb50a-9f79-43c4-ab98-c754885dbda9"
x = np.random.randint(0, 10, (8, 10))

x

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Vrkpn1d4qBNF" outputId="f0f3154b-bf1f-4acc-d14d-9b1bd1dde7b4"
# 平均値
x.mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Y_50qftbhs34" outputId="f0a8f5db-8c8e-465a-fd5c-4f788c0aae14"
# 分散
x.var()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="OwpUSNzeqMLq" outputId="f8abd453-55cf-4465-cd41-7c8d94982389"
# 標準偏差
x.std()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="xw_9lddYqcHF" outputId="5230bc25-d738-4220-9c83-f73dbd067765"
# 最大値
x.max()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="zls9oBfsqfJn" outputId="873c3f93-0116-42f7-df85-4b50bce5b762"
# 最小値
x.min()

# + [markdown] colab_type="text" id="wLLCGMeLhs4I"
# ここで、`x` は 2 次元配列なので、各次元に沿ったこれらの統計値の計算も行えます。
# 例えば、最後の次元内だけで平均をとると、8 個の平均値が得られるはずです。
# 平均を計算したい軸（何次元目に沿って計算するか）を `axis` という引数に指定します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="uvfzeCY3hs4I" outputId="b68937c7-a415-42a4-b7e5-1f6b79df82de"
x.mean(axis=1)

# + [markdown] colab_type="text" id="4be3BgZ6hs4M"
# これは、以下のように 1 次元目の値の平均を計算していったものを並べているのと同じことです。
# （ゼロベースインデックスで考えています。`x` の形は `(8, 10)` なので、0 次元目のサイズが 8、1 次元目のサイズが 10 です。）

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="sjqqYnSths4N" outputId="898b5fbd-4e31-4717-8f0b-d5373f214c80"
np.array([
    x[0, :].mean(),
    x[1, :].mean(),
    x[2, :].mean(),
    x[3, :].mean(),
    x[4, :].mean(),
    x[5, :].mean(),
    x[6, :].mean(),
    x[7, :].mean(),
])

# + [markdown] colab_type="text" id="rulZyy5osZxR"
# ## NumPy を用いた重回帰分析
#
# [単回帰分析と重回帰分析](https://tutorials.chainer.org/ja/07_Regression_Analysis.html)の章で説明した重回帰分析を NumPy を用いて行いましょう。
#
# 4 つのデータをまとめた、以下のようなデザイン行列が与えられたとします。

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="0grPocVNvn3P" outputId="1727f10c-dc5c-4864-937b-61a4e0fb4581"
# Xの定義
X = np.array([
    [2, 3],
    [2, 5],
    [3, 4],
    [5, 9],
])

X

# + [markdown] colab_type="text" id="1T4NGECihs4b"
# 4 章の解説と同様に、切片を重みベクトルに含めて扱うため、デザイン行列の 0 列目に 1 という値を付け加えます。

# + colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="aSSfzZoahs4d" outputId="0d1dbc3e-12cd-4bf5-bfca-fc8bb256a79e"
# データ数（X.shape[0]) と同じ数だけ 1 が並んだ配列
ones = np.ones((X.shape[0], 1))

# concatenate を使い、1 次元目に 1 を付け加える
X = np.concatenate((ones, X), axis=1)

# 先頭に 1 が付け加わったデザイン行列
X

# + [markdown] colab_type="text" id="DQ9Ybyrphs4g"
# また、目標値が以下で与えられたとします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="eU0UlmKrvxVK" outputId="7fb37396-09f2-427b-f52f-769b587f1b4c"
# t の定義
t = np.array([1, 5, 6, 8])

t

# + [markdown] colab_type="text" id="UX4hYhjtwBlK"
# 重回帰分析は、正規方程式を解くことで最適な 1 次方程式の重みを決定することができました。
# 正規方程式の解は以下のようなものでした。
#
# $$
# {\bf w} = ({\bf X}^{{\rm T}}{\bf X})^{\rm -1}{\bf X}^{\rm T}{\bf t}
# $$
#
# これを、4 つのステップに分けて計算していきます。
#
# まずは、${\bf X}^{\rm T}{\bf X}$ の計算です。ndarrayに対して `.T` で転置した配列を得られます。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="r_s8lxK9wvFW" outputId="5ea9564a-8e50-4db9-f334-1d38eb14f40d"
# Step 1
xx = np.dot(X.T, X)

xx

# + [markdown] colab_type="text" id="1x4UD4Tghs4v"
# 次に、この逆行列を計算します。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="V4ZpY_CLxuRX" outputId="1d4c9b86-c4e8-487c-afe7-6d5b09eb2508"
# Step 2
xx_inv = np.linalg.inv(xx)

xx_inv

# + [markdown] colab_type="text" id="s40OeZJ0yJw5"
# 逆行列の計算は `np.linalg.inv()` で行うことができます。
#
# 次に、${\bf X}^{\rm T}{\bf t}$ の計算をします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="YR6kbSD1x3XU" outputId="3ed492c2-175e-4e0b-eeea-8c4d18758804"
# Step 3
xt = np.dot(X.T, t)

xt

# + [markdown] colab_type="text" id="94huqVEDhs40"
# 最後に、求めた `xx_inv` と `xt` を掛け合わせます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="oBBA-NRbybHK" outputId="e7e35e3c-2566-473e-88dd-09a0ef7790e1"
# Step 4
w = np.dot(xx_inv, xt)

w

# + [markdown] colab_type="text" id="XsIpR4_ehs43"
# **以上の計算は、以下のように 1 行で行うこともできます。**

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="SXPsTsIxhs43" outputId="c862c977-b762-480d-bcd1-cece5355ff10"
w_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)

w_

# + [markdown] colab_type="text" id="k9xtcA31mXD-"
# 実際には逆行列を陽に求めることは稀で、連立一次方程式を解く、すなわち逆行列を計算してベクトルに掛けるのに等しい計算をひとまとめに行う関数 `numpy.linalg.solve` を呼ぶ方が速度面でも精度面でも有利です。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="rL5mO_Srm7Xe" outputId="8cf416e8-031b-4269-e8c2-7406ed3c5630"
w_ = np.linalg.solve(X.T.dot(X), X.T.dot(t))

w_

# + [markdown] colab_type="text" id="SxZkwFz8hs45"
# 数式を NumPy による配列の計算に落とし込むことに慣れていくには少し時間がかかりますが、慣れると少ない量のコードで記述できるだけでなく、高速に計算が行なえるため、大きな恩恵があります。

# + [markdown] colab_type="text" id="ky_Jm7KjyEsx"
# <hr />
#
# <div class="alert alert-info">
# **注釈 1**
#
# ライブラリとは、汎用性の高い複数の関数やクラスなどを再利用可能な形でひとまとまりにしたもので、Python の世界では**パッケージ**とも呼ばれます。また、Python で関数やクラスの定義、文などが書かれたファイルのことを**モジュール**と呼び、パッケージはモジュールが集まったものです。
#
# [▲上へ戻る](#ref_note1)
# </div>
#
# <div class="alert alert-info">
# **注釈 2**
#
# NumPy には matrix というクラスも存在しますが、本チュートリアルでは基本的に多次元配列を表す ndarray をベクトルや行列を表すために用います。
#
# [▲上へ戻る](#ref_note2)
# </div>
#
# <div class="alert alert-info">
# **注釈 3**
#
# これは、その多次元配列が表すテンソルの**階数（rank、以下ランク）**と対応します。
#
# [▲上へ戻る](#ref_note3)
# </div>
#
# <div class="alert alert-info">
# **注釈 4**
#     
# 「次元のサイズ」と言った場合はその次元の大きさを意味し、配列の `size` 属性とは異なるものを指しています。
#
# [▲上へ戻る](#ref_note4)
# </div>
#
# <div class="alert alert-info">
# **注釈 5**
#     
# 末尾次元（trailing dimension）とは、その配列の形を表すタプルの一番最後の値のことを指します。
#
# [▲上へ戻る](#ref_note5)
# </div>
