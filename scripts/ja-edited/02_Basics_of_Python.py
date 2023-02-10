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

# + [markdown] colab_type="text" id="D3QflLv0qdiy"
# # Python 入門
#
# 本章では、プログラミング言語 Python の基礎的な文法を学んでいきます。
# 次章以降に登場するコードを理解するにあたって必要となる最低限の知識について、最短で習得するのが目標です。
# より正確かつ詳細な知識を確認したい場合には、[公式のチュートリアル](https://docs.python.jp/3/tutorial/index.html)などを参照してください。
#
# Pythonにはバージョンとして 2 系と 3 系の 2 つの系統があり、互換性のない部分もあります。
# 本チュートリアルでは、3 系である **Python 3.6** 以上を前提とした解説を行っています。
#
# ## Python の特徴
#
# プログラミング言語には、Python 以外にも C 言語や Java、Ruby、R のように様々なものがあります。それぞれの言語がすべての用途に適しているわけではなく、しばしば用途によって得手不得手があります。
#
# 本チュートリアルでは基本的に Python というプログラミング言語を扱います。
# その理由は、Python はデータ解析・機械学習のためのライブラリが充実しており、データ解析や機械学習の分野で最もよく使われている言語だからです。
# また、Web アプリケーションフレームワークの開発も活発で、データ解析だけでなく Web サービス開発まで同じ言語で統一して行える点も魅力です。
#
#
# 本チュートリアルのテーマである Chainer も Python 向けに開発されています。

# + [markdown] colab_type="text" id="unYvpNZCv9aF"
# さらには、初学者にとっても学びやすい言語です。
# 初学者がプログラミングを学び始めるときにつまづきがちな難しい概念が他の言語と比べ多くなく、入門しやすい言語といえます。
#
# まとめると、Python には
#
# - データ解析や機械学習によく使われている
# - Web アプリケーションの開発などでもよく使われている
# - 初学者がはじめやすい言語
#
# のような魅力があります。

# + [markdown] colab_type="text" id="jZNTuBQ54BSu"
# ### 文法とアルゴリズム
#
# プログラミングによってある特定の処理をコンピュータで自動化したい場合、**文法**と**アルゴリズム**の 2 つを理解しておく必要があります。
#
# プログラムでは、まずはじめにコンピュータに命令を伝えるためのルールとなる**文法**を覚える必要があります。
# 文法を無視した記述があるプログラムは、実行した際にエラーとなり処理が停止します。そのため、文法はしっかりと理解しておく必要があります。
#
# ただし、文法さえ理解していれば十分かというとそうではありません。一般的に、初学者向けのプログラミングの参考書では、この文法だけを取り扱うことも多いのですが、コンピュータに処理を自動化させることが目的であれば、文法だけでなく**アルゴリズム**も理解する必要があります。アルゴリズムとは、どういう順番でどのような処理をしていくかの一連の手順をまとめたものです。
#
# この章では、Python の文法について紹介し、機械学習やディープラーニングで必要となるアルゴリズムについてはこれ以降の章で紹介します。
#
# ここでは以下 4 つの文法に主眼を置きながら説明していきます。
#
# - 変数
# - 制御構文
# - 関数
# - クラス

# + [markdown] colab_type="text" id="hO4YLiKjIfDi"
# ## 変数
#
# **変数 (variable)** とは、様々な値を格納することができる、**名前がついた入れ物**です。
# この変数に値を格納したり、更新したりすることで、計算結果を一時的に保持しておくことができます。
#
# ### 代入と値の確認
#
# それでは、`a` という名前の変数に、`1` を**代入**してみましょう。

# + colab={} colab_type="code" id="SjpDQJn-3Rct"
a = 1

# + [markdown] colab_type="text" id="aitz3Hri_a-n"
# 代入は `=` の記号を用います。
# 数学的には `=` は等しいという意味を持ちますが、Python では**「左辺の変数に、右辺の値を代入する」**という意味になります。
#
# Jupyter Notebook 上では、変数名だけ、もしくは変数名を最後の行に記述したセルを実行すると、値を確認することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6SbAfM5Rv9aM" outputId="e8173162-1c03-418e-f01f-3785b6b255e9"
a

# + [markdown] colab_type="text" id="8b0i8oelv9aR"
# このように、変数に格納されている値を確認することができました。
# また、値を確認するための他の方法として、`print()` と呼ばれる**関数 (function)** を使用することもできます。
# 関数について詳しくは後述しますが、`print()` のように Python には予め多くの関数が定義されています。 そのような関数を**組み込み関数 (built-in function)** といいます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="oTHXMTjiv9aR" outputId="fc9f6804-a059-48f1-ae0a-45ca25eecc86"
print(a)

# + [markdown] colab_type="text" id="nGNlqHW_v9aV"
# 変数につける名前は、コードを書く人が自由に決めることができます。
# ただし、わかりやすい名前をつけることがとても大切です。
# 例えば、人の名前を格納するための変数が `a` という変数名だと、それがどのような使われ方をするのかを容易に類推することができません。
# `name` という名前であれば、ひと目で見て何のための変数かが分かるようになります。
# これは、自分のコードを読む他人や、未来の自分にとってコードを理解するための大きな手がかりとなります。

# + [markdown] colab_type="text" id="BQhk7-XfDFhS"
# ### コメント
#
# Python では、`#` の後からその行の終わりまでに存在する全ての文字列は無視されます。
# この `#` の後ろに続く部分を**コメント (comment)**と呼び、すでに書かれたコードをコメントにすることを**コメントアウト (comment out)**と言います。
# コメントは、コード中に変数の意味や処理の意味をコードを読む人に伝えるためによく使われます。
#
# Jupyter Notebook のコードセルに書かれたコードを行ごとコメントアウトしたい場合は、その行を選択した状態で `Ctrl + /` を入力することで自動的に行の先頭に `#` 記号を挿入することができます。複数行を選択していれば、選択された複数の行が同時にコメントアウトされます。また、コメントアウトされている行を選択した状態で同じキー入力を送ると、コメントアウトが解除されます。これを**アンコメント (uncomment)**と呼ぶこともあります。
#
# 下のセルを実行してみましょう。

# + colab={} colab_type="code" id="YK_XjChXDXmH"
# この行及び下の行はコメントアウトされているため実行時に無視されます
# print(a)

# + [markdown] colab_type="text" id="_xAqta5WIfGJ"
# `print(a)` が書かれているにも関わらず、何も表示されませんでした。
# これは、`print(a)` 関数が書かれた行がコメントアウトされていたためです。

# + [markdown] colab_type="text" id="I_Z6hhts_v4c"
# ### 変数の型
#
# プログラミングで扱う値には種類があります。
# Python では、**整数 (integer)**、**実数 (real number)**、**文字列 (string)** などが代表的な値の種類です。
# それぞれの種類によって、コンピュータ内での取扱い方が異なり、この種類のことは一般に**型 (type)** と呼びます。
#
# 例えば、整数、実数、そして文字列をそれぞれ別々の変数に代入するコードは以下のとおりです。

# + colab={} colab_type="code" id="FChuhC8Jv9aa"
a = 1

# + colab={} colab_type="code" id="qCdjuQqNv9ac"
b = 1.2

# + colab={} colab_type="code" id="VLMovMgnv9ae"
c = 'Chainer'

# + [markdown] colab_type="text" id="DwMajDvAv9ag"
# コンピュータの中での取り扱い方は異なりますが、Python では**変数の型を自動的に決定する**ため、初めのうちはあまり気にする必要はありません。
# ただし、違う型同士の演算では場合によってエラーが発生するなどの問題が生じるため、簡単に型の特徴は把握しておく必要があります。
#
# まずは、上記の `a`, `b`, `c` の型を確認する方法を紹介します。
# 型の確認は `type()` という組み込み関数を使用します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="RUflpSu6v9ag" outputId="3219c3be-c030-478f-ef28-b2da097fc8f4"
type(a)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="YBtHAigUv9aj" outputId="36b7795d-d590-4b51-c02e-b3aa656d6bd5"
type(b)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6PaQmlMCv9an" outputId="a0b6ebc3-0871-47f6-b8e1-46937b23d623"
type(c)

# + [markdown] colab_type="text" id="gatuDQAyv9aq"
# `a` は `int` という整数の型をもつ変数であり、`b` は `float` という実数の型をもつ変数です。
# この `float` という型の名前は、コンピュータ内で実数を扱うときの形式である**浮動小数点数 (floating-point number)** に由来しています。
#
# `c` は `str` という文字列の型をもつ変数であり、値を定義するにはシングルクォーテーション `' '` もしくはダブルクォーテーション `" "` で対象の文字列をくくる必要があります。

# + [markdown] colab_type="text" id="Chdmx6HGIfEV"
# Python では、`.` を含まない連続した数字を `int`、直前・直後も含め `.` が含まれる連続した数字を `float` だと自動的に解釈します。
# 例えば、`7` や `365` は `int` ですが、`2.718`、`.25`、`10.` などは `float` になります。
#
# 実数の `0` は `0.0` とも `.0` とも `0.` とも書くことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="sUKUmRgwIfEY" outputId="797e97a5-ea05-4a08-ac99-d6b7cbd39f49"
type(0)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="VOiscn8OIfEc" outputId="f84135ab-0652-4a21-cfad-f93518575455"
type(0.)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="442-_TpmIfEf" outputId="3516a8d3-02b6-4b2d-b366-aec051d785e5"
type(.0)

# + [markdown] colab_type="text" id="by0V-W7nIfEj"
# 例えば、実数の `5` は以下のように書くことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="uUYCOBaAIfEj" outputId="00c3e638-e7dd-4d7a-affa-863d9407c9e0"
type(5.0)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="y26NMdQ2IfEo" outputId="cc8fe9ce-1a65-444c-cc59-3d11de8862b0"
type(5.)

# + [markdown] colab_type="text" id="g3BdGDJaIfEr"
# 一方、`.5` と書くと、これは `0.5` の略記と解釈されることに注意してください。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="aXpxR7WmIfEs" outputId="f6ade488-902d-4d96-9cc0-e0a96caca7f6"
type(.5)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="enBKbL9fIfEx" outputId="992db59f-1989-4d2d-db0d-241aa434cc83"
print(.5)

# + [markdown] colab_type="text" id="VsdsnV9uiQev"
# ### 算術演算子
#
# 様々な計算を意味する**演算子**と呼ばれる記号があります。
# はじめに紹介するのは**算術演算子 (arithmetic operator)** と呼ばれるもので、 2 つの変数または値を取り、 1 つの演算結果を返します。
#
# 代表的な演算として**四則演算（加算・減算・乗算・除算）**があります。
# 四則演算に対応する演算子として、Python では以下の記号が用いられます。
#
# | 演算 | 記号 |
# |------|------|
# | 加算（足し算） | `+` |
# | 減算（引き算） | `-` |
# | 乗算（掛け算） | `*` |
# | 除算（割り算） | `/` |
#
# 具体例を見ながら使い方を説明します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="MBaRZ2RkBWbd" outputId="744c489e-75bd-4fbb-f6c1-7e73d751ff07"
# 整数と整数で加算 -> 結果は整数
1+1

# + [markdown] colab_type="text" id="DE56EVRvIfFJ"
# このように、演算子を使う際には、**記号の両側に値を書きます。**
# このとき、演算子の両側にひとつずつ空白を空けることが多いです。
# 文法的な意味はありませんが、コードが読みやすくなります。
# この空白は Python のコーディング規約である [PEP8](https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator) でも推奨されています。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="BSsWGE_Ev9bH" outputId="3a1c9e29-c5de-4d34-b3b3-3a240378e4bd"
1 + 1

# + [markdown] colab_type="text" id="ai1hzuv6v9bJ"
# 値が代入されている変数との演算も下記のように行うことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="yvA4BP5hIfFJ" outputId="cedf23b1-33b6-4f96-c655-9b64346e755c"
a + 2

# + [markdown] colab_type="text" id="y6xkeITrv9bL"
# また、`int` と `float` は異なる型同士ですが、計算を行うことができます。
# `int` と `float` の演算結果の型は `float` になります。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="IjU6vwcdIfFM" outputId="21f73fdf-108e-4988-f2b4-4cfcffc219bf"
# 整数と実数で加算 -> 結果は実数
a + b

# + [markdown] colab_type="text" id="PwsK4SxPIfFN"
# 他の演算子の例を示します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="hL5QmnGgBYNF" outputId="d408993d-3a50-4e49-9e40-f15696984208"
# 整数と整数で減算 -> 結果は整数
2 - 1

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="huifgJVkv9bR" outputId="1b6ddcff-4a15-4306-b615-cc27c0ccb0b2"
# 実数と整数で減算 -> 結果は実数
3.5 - 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="v82yE4keBZeo" outputId="40c8b70c-1e5a-4649-c5d4-141a9cefc76e"
# 整数と整数で乗算 -> 結果は整数
3 * 5

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="kYYOMXTHv9bX" outputId="a1fde9ed-7a8f-4184-f2e4-86e1a76b6c3d"
# 実数と整数で乗算 -> 結果は実数
2.5 * 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="KHRGnOcNBi05" outputId="a988ca7c-861f-4529-b243-c195573c0e2c"
# 整数と整数で除算 -> 結果は実数
3 / 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="NxnIvty-v9bd" outputId="d1057e6c-490d-4dfc-fea6-53b6121bda09"
# 整数と整数で除算 -> 結果は実数
4 / 2

# + [markdown] colab_type="text" id="TDJRJR5vIfFe"
# Python 3 では、 `/` 記号を用いて除算を行う場合、除数（割る数）と被除数（割られる数）が整数であっても、計算結果として実数が返ります。
# 計算結果として実数を返す除算のことを特に、**真の除算 (true division)** と言います。
# 一方、商（整数部分）を返すような除算演算子として、 `//` 記号が用意されています。 `/` 記号を 2 回、間を空けずに繰り返します。計算結果として商を返す除算のことを、 **切り捨て除算 (floor division)** と呼びます。
# 商を計算したい場合に便利な演算子であるため、こちらも覚えておきましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="PFwX_4_pBbY0" outputId="260dd494-deb0-49c6-edb6-19d1662137e7"
# 整数と整数で切り捨て除算 -> 結果は整数
3 // 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="nCyFpJHKv9bh" outputId="32099b02-2f65-46c5-89d9-343396552ed9"
# 整数と整数で切り捨て除算 -> 結果は整数
4 // 2

# + [markdown] colab_type="text" id="Mywi7srHv9bj"
# また、ここで注意すべき点として、整数や実数と文字列の演算は基本的にエラーになります。

# + colab={"base_uri": "https://localhost:8080/", "height": 172} colab_type="code" id="ia8mAAr5v9bj" outputId="38cd2f68-c08d-4abe-c8bf-300d8c8b1154"
# error
a + c

# + [markdown] colab_type="text" id="T24iNnwSv9bk"
# **エラーメッセージを読みましょう。**
#
# > TypeError: unsupported operand type(s) for +: 'int' and 'str'
#
# と言われています。「+ にとって int と str はサポートされていない被作用子（+ が作用する対象のこと。operand）です」と書かれています。「int に str を足す」ということはできないというわけです。
#
# このようにエラーメッセージからは自分のミスに関する情報を得ることができます。
# 何を間違えたかはエラーをもとに調べれば大抵わかりますから、まずはエラーについて調べることを心がけましょう。
#
# `int` もしくは `float` と、 `str` の間の加算、減算、除算では上記のエラーが生じます。
# ただし、`str` と `int` の**乗算**は特別にサポートされており、計算を実行することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="oKJmWQKcv9bl" outputId="513c4f7e-081e-4b3f-826a-39e8e5885dff"
# str と int で乗算
c * 3

# + [markdown] colab_type="text" id="xPX4dcGUv9bm"
# 上のコードは、`c` という文字列を `3` 回繰り返す、という意味になります。

# + [markdown] colab_type="text" id="mmlvVMcIv9bm"
# `str` 同士は足し算を行うことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="DOK7H8LJv9bn" outputId="5654572a-9d44-423a-9ada-da324b5f7dfa"
name1 = 'Chainer'
name2 = 'チュートリアル'

name1 + name2

# + [markdown] colab_type="text" id="QoCq87hkv9bq"
# 整数と文字列を連結したいこともあります。
# 例えば、`1` という整数に、 `'番目'` という文字列を足して `'1番目'` という文字列を作りたいような場合です。
# その場合には、型を変換する**キャスト (cast)** という操作をする必要があります。
#
# 何かを `int` にキャストしたい場合は `int()` という組み込み関数を使い、`str` にキャストしたい場合は `str()` という組み込み関数を使います。では、`1` という整数を `str` にキャストして、 `'番目'` という文字列と足し算を行ってみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="H8E1Puhmv9bq" outputId="31d58419-7b97-4bc6-f923-79f2c8fd9edb"
1

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TlnTtXUnv9br" outputId="b970fbfe-4151-42f9-ad44-0e607f912e02"
type(1)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="EK9fF95av9bt" outputId="a5ef7604-c6ed-4ffd-e19c-39b39496c752"
str(1)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="JxGEBOZzv9bw" outputId="cca1f61a-82c9-4f12-9921-97068ab3a6d0"
type(str(1))

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="2boojYfbv9by" outputId="9ac453cd-d210-4f5c-fcac-1502f0b69482"
str(1) + '番目'

# + [markdown] colab_type="text" id="RzvUMnTQv9bz"
# また、`+=` や `-=` もよく使います。
# これは、演算と代入を合わせて行うもので、**累積代入文 (augmented assignment statement)** と呼ばれます。
#
# 下記に示すとおり、`+=` では左辺の変数に対して右辺の値を足した結果で、左辺の変数を更新します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="g7g0hOrOv9b0" outputId="914559f8-251d-4d5c-9edd-6cc9a778c484"
# 累積代入文を使わない場合
count = 0
count = count + 1
count

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="bdgvqEjZv9b2" outputId="0cbef171-cbc1-4cb8-b965-1f52fcd005a6"
# 累積代入文を使う場合
count = 0
count += 1
count

# + [markdown] colab_type="text" id="AZ7QMIt-v9b3"
# 四則演算の全てで累積代入文を利用することができます。
# つまり、`+=`, `-=`, `*=`, `/=` がそれぞれ利用可能です。

# + [markdown] colab_type="text" id="36GMdun-IfFn"
# Python には、他にも幾つかの算術演算子が用意されています。
# 例えば以下の演算子です。
#
# | 演算 | 記号 |
# |------|------|
# | 累乗 | `**` |
# |  剰余　 | `%` |
#
# `**` を使うと、$2^3$ は以下のように記述することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="cK6zHxguIfFo" outputId="eb9bdd74-8294-49f5-aae4-a9566f3d7ddf"
# 累乗
2 ** 3

# + [markdown] colab_type="text" id="g8RG0jTDv9b6"
# `%` を使って、`9` を `2` で割った余りを計算してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="CbsiwPcev9b6" outputId="27688f90-fef0-4d2d-8633-960ea0e8b128"
# 剰余
9 % 2

# + [markdown] colab_type="text" id="Z62QzE-4Bc4a"
# ### 比較演算子
#
# 比較演算子は、2 つの値の比較を行うための演算子です。
#
# | 演算 | 記号 |
# |------|------|
# | 小なり | `<` |
# | 大なり | `>` |
# | 以下 | `<=` |
# | 以上 | `>=` |
# | 等しい | `==` |
# | 等しくない | `!=` |
#
# 比較演算子は、その両側に与えられた値が決められた条件を満たしているかどうか計算し、満たしている場合は `True` を、満たしていない場合は `False` を返します。
# `True` や `False` は、**ブール (bool) 型**と呼ばれる型を持った値です。
# ブール型の値は `True` もしくは `False` の 2 つしか存在しません。
#
# いくつかの比較演算子の計算例を示します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="5AXOptIWCN53" outputId="2525fafa-f2dd-40b6-adaa-9af505b059c4"
1 < 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="ETSRMdkDv9b-" outputId="fb2a62ee-87a1-488a-9a77-e99854e5a083"
# 型の確認
type(1 < 2)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="wR2OPEwiCP5_" outputId="bd6701e6-b948-4ebb-f852-12ac81244631"
2 == 5

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TfEDslvKCQ7o" outputId="629c30b4-ab73-49be-e8dc-e9bd78ba460b"
1 != 2

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="W1v_F8-7CWI-" outputId="789351f8-d4a5-41a7-8732-72f5ab885abd"
3 >= 3

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="XapxlGSYCSOU" outputId="7f4fac18-bed4-490d-cb25-598737c2f7e2"
'test' == 'test'

# + [markdown] colab_type="text" id="aBzVaMJ1IfF7"
# 等しいかどうかを判定する比較演算子 `==` を使う際は、代入演算子 `=` と間違えないように気をつけてください。

# + [markdown] colab_type="text" id="VsBRE0Wyv9ca"
# ## 複合データ型
#
# これまでは `a = 1` のように 1 つの変数に 1 つの値を代入する場合を扱ってきましたが、複数の値をまとめて取り扱いたい場面もあります。
# Python では複数の変数や値をまとめて扱うのに便利な、以下の 3 つの複合データ型があります。
#
# - リスト (list)
# - タプル (tuple)
# - 辞書 (dictionary)

# + [markdown] colab_type="text" id="S1TLaajMDtmr"
# ### リスト
#
# 複数の変数を `,` （カンマ）区切りで並べ、それらの全体を `[ ]` で囲んだものを **リスト (list)** と言います。
# リストに含まれる値を**要素**と呼び、整数の**インデックス** （要素番号）を使ってアクセスします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="MJ7ZYj82D-a1" outputId="970de678-1f56-4078-8f10-c7042c0fb8e5"
# リスト型の変数を定義
numbers = [4, 5, 6, 7]

# 値の確認
print(numbers)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="HH3NT-y4EC4Z" outputId="cbcc163e-040f-4878-d3f5-65b6659bffa3"
# 型の確認
type(numbers)

# + [markdown] colab_type="text" id="Rk_jY_TTIfGY"
# `numbers` には 4 つの数値が入っており、**要素数** は 4 です。
# リストの要素数は、リストの**長さ (length)** とも呼ばれ、組み込み関数の `len()` を用いて取得することができます。
# `len()` はよく使う関数であるため、覚えておきましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="DK6aH0AaoELi" outputId="76d0dab8-bd1b-4c86-95b8-e25d6cca02c5"
# 要素数の確認
len(numbers)

# + [markdown] colab_type="text" id="RiJeRUdrEEUq"
# リストの各要素へアクセスする方法はいくつかあります。
# 最も簡単な方法は `[]` を使ってアクセスしたい要素番号を指定して、リストから値を取り出したり、その位置の値を書き換えたりする方法です。
# ここで、注意が必要な点として、Python では先頭の要素のインデックス番号が `0` である点があります。
# インデックス番号 `1` は 2 番目の要素を指します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="DeCsf_L1Em29" outputId="4cfc90bc-62e4-466d-ab0d-0a29877decef"
# 先頭の要素にアクセス
numbers[0]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="ox-Ma4yv5xqO" outputId="7c6ee95d-c6cd-472d-9ea5-c4a44e8d9f0f"
# 先頭から3番目の要素にアクセス
numbers[2]

# + colab={} colab_type="code" id="ve-ETFIrv9cg"
# 2 番目の要素を書き換え
numbers[1] = 10

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="AXpG-tqNv9cg" outputId="1fa368f6-7164-4896-89fd-981d033adaa3"
# 値の確認
numbers

# + [markdown] colab_type="text" id="kOVYiVOwIfGW"
# また、インデックスに負の値を指定すると、末尾からの位置となります。
# 要素番号 `-1` で最後の要素を参照することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="WSaIdvb4IfGX" outputId="5bbb1cee-081d-4a68-a918-ad275b3b1172"
# 末尾の要素にアクセス
numbers[-1]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="jP5fkl3V54WL" outputId="aaf2ac71-f21a-4c41-aab4-337e25f54331"
# 末尾から3番目の要素にアクセス
numbers[-3]

# + [markdown] colab_type="text" id="DLzLjTQ8EoG9"
# 次に、リストから一度に複数の要素を取り出す操作である**スライス (slice)** を紹介します。
# `開始位置:終了位置` のようにコロン `:` を用いてインデックスを範囲指定し、複数の部分要素にアクセスします。
# このスライスの処理は、この後の章でも多用するため、慣れておきましょう。
#
# 例えば、先頭から 2 つの要素を取り出したい場合、以下のように指定します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="MPI2H38DEy2N" outputId="623cb4f8-54fc-4f34-924c-7da10dacdeef"
numbers[0:2]

# + [markdown] colab_type="text" id="gK6_JXXTE2AY"
# `開始位置:終了位置` と指定することで、開始位置から**終了位置のひとつ手前**までの要素を抽出します。 
# 終了位置に指定したインデックスの値は含まれないことに注意してください。
#
# また、指定する開始番号が `0` である場合、以下のような略記がよく用いられます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="hBo2_C5aFPHR" outputId="1b249507-b615-4adf-dd4b-3c399799ba88"
numbers[:2]

# + [markdown] colab_type="text" id="Qqem8sujFQVP"
# このように、先頭のインデックスは省略することができます。
# このような記法を使う場合は、終了位置を示す数字を**取り出したい要素の個数**と捉えて、**先頭から 2 つを取り出す**操作だと考えると分かりやすくなります。
#
# 同様に、ある位置からリストの末尾までを取り出す場合も、終了位置のインデックスを省略することができます。
# 例えば、2 個目の要素から最後までを取り出すには以下のようにします。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="gItFJqoUFYzX" outputId="2bc2f7e8-b6cb-4bbb-8245-ec8835e87447"
numbers[1:]

# + [markdown] colab_type="text" id="RRlf3vqOv9cq"
# この場合は、取り出される要素の個数は `len(numbers) - 1` 個となることに注意してください。  
# 以上から、`numbers[:2]` と `numbers[2:]` は、ちょうど 2 個目の要素を境に `numbers` の要素を 2 分割した前半部分と後半部分になっています。
# ここで、インデックスが 2 の要素自体は**後半部に含まれる**ということに注意してください。
# また、開始位置も終了位置も省略した場合は、すべての要素が選択されます。
#
# 一見ややこしい挙動のようですが、ここで下画像のようにインデックスは要素の間にふられているとすると理解しやすくなります。  
# 例えば `numbers[:2]` は「2の位置"まで"」だから2個目の要素は含まないというわけです。  
# ![indexのイメージ](./images/02/02_indexing.jpg)
#
# ちなみにこれは[Pythonの公式ドキュメント](https://docs.python.org/ja/3/tutorial/introduction.html#strings)で紹介されている覚え方です。
# -

# 全ての要素を選択する場合、下記のように書くこともできます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="nav8WoxNv9cq" outputId="b35c3b9d-f123-42a4-9d5b-505a31be278e"
numbers[:]

# + [markdown] colab_type="text" id="9qPrP6sYv9cs"
# 現状では、`numbers[:]` と `numbers` の結果が同じであるため、どのように使用するか疑問に思われるかも知れません。  
# しかし、後の章では NumPy というライブラリを用いてリストの中にリストが入ったような**多次元配列 (multidimensional array)** を扱っていきます。  
# そして多次元配列を用いて行列を表す場合には、`0 列目のすべての値`を抽出するために `[:, 0]` のような記法を用いるケースが登場します。  
# これは Python 標準の機能ではありませんが、Python 標準のスライス表記を拡張したものになっています。
#

# + [markdown] colab_type="text" id="AyoiOU5OFfET"
# リストは数値以外に、文字列を扱うこともでき、また複数の型を同一のリスト内に混在させることもできます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="9JglgRiWFyo8" outputId="f8700d40-519f-4e92-db32-60fcadc145c0"
# 文字列を格納したリスト
array = ['hello', 'world']
array

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Kcf8MyrBv9cu" outputId="c5f989f4-138a-4a1c-ac4a-60760f318618"
# 複数の型が混在したリスト
array = [1, 1.2, 'Chainer']
array

# + [markdown] colab_type="text" id="9fKT2sjNv9cu"
# リストにリストを代入することもできます。
# また、Python 標準のリストでは入れ子になったリスト内の要素数がばらばらでも問題ありません。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Q8JxfHS8v9cv" outputId="572e4d48-aae1-46e9-cf6a-f2fbf1d68ac4"
array = [[1, 1.2, 'Chainer', True], [3.2, 'Tutorial']]
array

# + [markdown] colab_type="text" id="5lYy-LP6F3wP"
# リストを使う際に頻出する操作として、**リストへの値の追加**があります。
# リスト型には `append()` というメソッドが定義されており、これを用いてリストの末尾に新しい値を追加することができます。
#
# 上記の `array` に値を追加してみましょう。

# + colab={} colab_type="code" id="RmgtEN8Lv9cw"
# 末尾に 2.5 を追加
array.append(2.5)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="2E8q5y94v9cx" outputId="c2993fd6-a0f4-4786-b61a-f5462cae0817"
# 値の確認
array

# + [markdown] colab_type="text" id="y5dQJA57v9cz"
# また、今後頻出する処理として、**空のリスト**を定義しておき、そこに後段の処理の中で適宜新たな要素を追加していくという使い方があります。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="00AEmnpxGPky" outputId="7f95e646-3264-4f63-cb58-98fc6bd42e73"
# 空のリストを定義
array = []

# 空のリストに要素を追加
array.append('Chainer')
array.append('チュートリアル')

array

# + [markdown] colab_type="text" id="CV1072C3M7Qy"
# ### タプル
#
# **タプル (tuple)** はリストと同様に複数の要素をまとめた型ですが、リストとは異なる点として、定義した後に**中の要素を変更できない**という性質を持ちます。
#
# タプルの定義には `( )`を用います。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="mo4lGZN6NGAI" outputId="8234b24e-fd6d-4e77-f409-0681ad27c406"
# タプルを定義
array = (4, 5, 6, 7)
array

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="LEUgknZov9c3" outputId="96b964b2-97ef-4312-d552-ee24b5d47238"
# 型の確認
type(array)

# + [markdown] colab_type="text" id="9w2M0A02v9c3"
# タプルの定義する際に `( )` を使用したため、要素へのアクセスも `( )` を使うように感じるかもしれませんが、実際にはリストと同様 `[ ]` を使用します。  

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Fq7zIHMfNIXG" outputId="8cc0dcde-bfc6-4d74-cb73-6a716394bb98"
# 先頭の要素へアクセス
array[0]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="3XdB2S5pv9c4" outputId="c270d121-31e8-4b38-f1c7-0f15b49f4b1e"
# リストと同様、スライスも使用可能
array[:3]

# + [markdown] colab_type="text" id="KjdkYtSmNSl3"
# 先述の通り、タプルは各要素の値を変更することができません。
# この性質は、定数項などプログラムの途中で書き換わってしまうことが望ましくないものをまとめて扱うのに便利です。
#
# 実際に、タプルの要素に値の書き換えを行うとエラーが発生します。

# + colab={"base_uri": "https://localhost:8080/", "height": 172} colab_type="code" id="DRoToOjUNI51" outputId="3db3cb2f-673f-4463-dbe3-82ced4e09830"
# error
array[0] = 10

# + [markdown] colab_type="text" id="6aq_3yNZv9c6"
# `tuple` のように中身が変更できない性質のことを**イミュータブル (immutable)**であると言います。反対に、`list` のように中身が変更できる性質のことを**ミュータブル (mutable)**であると言います
#
# タプルも Chainer でデータセットを扱うときなどに頻出する型です。その性質と取り扱い方を覚えておきましょう。

# + [markdown] colab_type="text" id="nV8Lp5jLNO66"
# ### 辞書
#
# リストやタプルでは、複数の値をまとめて扱うことができました。
# そこで、定期テストの結果をまとめることを考えてみましょう。
#
# 例えば、数学 90 点、理科 75 点、英語 80 点だったという結果を `scores = [90, 75, 80]` とリストで表してみます。
# しかし、これでは**何番目がどの教科の点数に対応するか**、一見して分かりにくいと思われます。
#
# Python の `dict` 型は、**キー (key)** とそれに対応する**値 (value)** をセットにして格納することができる型であり、このようなときに便利です。
#
# リストやタプルでは、各要素にアクセスする際に整数のインデックスを用いていましたが、辞書ではキーでインデックス化されているため、整数や文字列など、色々なものを使って要素を指定することができます。
#
# 辞書は `{}` を用いて定義し、要素にアクセスする際には、リストやタプルと同様に `[ ]` を使用し、`[ ]` の中にキーを指定して対応する値を取り出します。
# -

# `[ ]` による要素へのアクセスは複数の要素を持つ型において一般的な記法なのがわかると思います。  
# 要素のどれにアクセスするのかを、リストやタプルではインデックスで、辞書ではキーで指定するわけです。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="NyxphI5fNMO8" outputId="c4d9a198-171f-4c0e-ec96-de8cdc161c0d"
# 辞書を定義
scores = {'Math': 90, 'Science': 75, 'English': 80 }
scores

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="M-V2XOEdNu2a" outputId="ee4bdf90-c4e5-4a92-a6d0-d79e3bcf2b1c"
# key が Math の value にアクセス
scores['Math'] 

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TA5V7SUBv9c8" outputId="90c16e6d-1de9-4931-efbe-b2164e075b6b"
# key に日本語を使用することも可能
scores = {'数学': 90, '理科': 75, '英語': 80}
scores

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="IL2y81t-v9c9" outputId="7ea8f85a-c3d2-4ad2-cda5-74bd524868f3"
scores['数学']

# + [markdown] colab_type="text" id="Jd_XpPMqfrSL"
# 他の人が定義した辞書に、**どのようなキーが存在するのか**を調べたいときがあります。
# 辞書には、そのような場合に使える便利なメソッドがいくつか存在します。
#
# - `keys()`: キーのリストを取得。`dict_keys` というリストと性質が似た型が返る
# - `values()`: 値のリストを取得。`dict_values` というリストと性質が似た型が返る
# - `items()`: 各要素の `(key, value)` のタプルが並んだリストを取得。`dict_items` というリストと性質が似た型が返る

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="FkJjSlbShKjK" outputId="93597067-1c5e-4ab0-ab5f-3301524f6bff"
# キーのリスト
scores.keys()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="YOAOTDtUhNBz" outputId="ad624c25-5b59-4544-dde2-94ec3adde7ea"
# 値のリスト
scores.values()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="mWA7opwAhVEa" outputId="26ee1d21-3011-49df-f1ff-e6188e698b1c"
# (キー, 値)というタプルを要素とするリスト
scores.items()

# + [markdown] colab_type="text" id="I9WEXgSZv9dC"
# `dict_keys`, `dict_values`, `dict_items` と新しい型が登場しましたが、これは辞書型特有の型であり厳密には標準のリストとは異なりますが、リストと性質の似た型であるという程度の認識で問題ありません。

# + [markdown] colab_type="text" id="qpoTmSY8bTDw"
# 辞書に要素を追加する場合は、新しいキーを指定して値を代入します。

# + colab={} colab_type="code" id="nDZ4RRcjnERu"
scores['国語'] = 85

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="P0HJVZJSnKal" outputId="700a1c1a-d175-40c9-c008-56dca29eb5f8"
scores

# + [markdown] colab_type="text" id="NIC2fwAknPf9"
# また、既に存在するキーを指定した場合には、値が上書きされます。

# + colab={} colab_type="code" id="teLZbAHQnUfN"
scores['数学'] = 95

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="kQESKoXgnYRN" outputId="1b2e3718-26d7-4d5e-fa90-e7fc4850795a"
scores

# + [markdown] colab_type="text" id="zTYLayT3qEzA"
# ## 制御構文
#
# 複雑なプログラムを記述しようとすると、繰り返しの処理や、条件によって動作を変える処理が必要となります。
# これらは**制御構文**を用いて記述します。
#
# ここでは最も基本的な制御構文を 2 つ紹介します。
#
# - 繰り返し (`for`, `while`)
# - 条件分岐 (`if`)
#
# Python の制御構文は、**ヘッダ (header)** と **ブロック (block)** と呼ばれる 2 つの部分で構成されています。
# これらを合わせて **複合文 (compound statement)** と呼びます。
#
# ![ヘッダーとブロック](images/02/02_05.png)

# + [markdown] colab_type="text" id="ciZc5_PCv9dJ"
# 上図に示すように、制御構文ではヘッダ行に `for` 文や `if-else` 句を記述し、行末に `:` 記号を書きます。次に、ヘッダ行の条件で実行したい一連の処理文を、ブロックとしてその次の行以降に記述していきます。その際、 **インデント (indent)** と呼ばれる空白文字を先頭に挿入することで、ブロックを表現します。同じ数の空白でインデントされた文がブロックとみなされます。
# Python では、インデントとして**スペース 4 つ**を用いることが推奨されています。

# + [markdown] colab_type="text" id="oELnTpERNwCQ"
# ### 繰り返し（for 文）
#
# 同じ内容のメールを宛名だけ個別に変えて、1000 人に一斉送信したい場合など、繰り返す処理を記述する制御構文である `for` を使います。
#
# ![for文](images/02/02_06.png)

# + [markdown] colab_type="text" id="_-50XwKOv9dJ"
# `for` 文の文法は上図のとおりです。
#
# **イテラブルオブジェクト (iterable object)** とは、反復可能オブジェクトのことであり、要素を一度に 1 つずつ返せるオブジェクトのことを指します。
# `range()` という組み込み関数を使うと、引数に与えた整数の回数だけ順番に整数を返すイテラブルオブジェクトを作ることができます。
# `range(5)` と書くと、0, 1, 2, 3, 4 という整数 5 つを順番に返すイテラブルオブジェクトになります。
#
# 後述しますが、このイテラブルオブジェクトとして、リストやタプルも指定することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 106} colab_type="code" id="f7Y4APgkON_7" outputId="8d3fedd9-b0af-4cf4-ec66-bffb78ce9569"
# 5回繰り返す
for i in range(5):
    print(i)

# + [markdown] colab_type="text" id="oLpTgIv4OPnt"
# 上記の例では、イテラブルオブジェクトが1 つずつ返す値を変数 `i` で受け取っています。
# 最初は `i = 0` から始まっていることに注意してください。
# 最後の値も、`5` ではなく `4` となっています。
# このように、`range()` に 1 つの整数を与えた場合は、その整数 - 1 まで 0 から 1 つずつ増えていく整数を順番に返します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="SUTjSLuKv9dK" outputId="be21ab95-21c5-4a56-8e73-35f157f13954"
# 繰り返し処理が終わった後の値の確認
i

# + [markdown] colab_type="text" id="4sMmj2Ewv9dL"
# Jupyter Notebook では変数名をコードセルの最後の行に書いて実行するとその変数に代入されている値を確認できましたが、for 文の中のブロックでは明示的に `print()` を使う必要があります。
# `print()` を用いないと、以下のように何も表示されません。

# + colab={} colab_type="code" id="2SlR9vcGv9dL"
# 変数の値は表示されない
for i in range(5):
    i

# + [markdown] colab_type="text" id="tEczQdb9v9dM"
# for 文を使って、0 から始まって 1 ずつ大きくなっていく整数順に取得し、これをリストのインデックスに利用すれば、リストの各要素に順番にアクセスすることができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="b6XXuQ_gOnxk" outputId="ad85251e-4bcf-4b1f-c9d0-0aead91f025e"
names = ['佐藤', '鈴木', '高橋']

for i in range(3):
    print(names[i])

# + [markdown] colab_type="text" id="msGhS1ISv9dN"
# つぎに、さらに汎用性の高いプログラムを目指します。
#
# 上記のコードに関して、汎用性が低い点として、`range(3)` のように `3` という値を直接記述していることが挙げられます。
# この `3` はリストの要素の数を意味していますが、リストの要素の数が変わると、このプログラムも書き換える必要があり、手間がかかったり、ミスが発生する原因となったりします。
#
# リスト内の要素の数は、組み込み関数である `len()` を用いて取得できるため、これを使用した汎用性の高いプログラムに書き換えましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="e4gbC4CIOzve" outputId="b5472139-d75b-46ac-de43-4ab365a4d14c"
len(names)

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="4i21QYMwPB1C" outputId="0dffef88-45d8-4025-c5bd-75a333a70e0c"
for i in range(len(names)):
    print('{}さん'.format(names[i]))

# + [markdown] colab_type="text" id="9wx-qJovPEC0"
# これでリストの要素数に依存しないプログラムにすることができました。

# + [markdown] colab_type="text" id="KIO9gkUnv9dR"
# また、リスト自体をイテラブルオブジェクトとして指定することにより、リスト要素数の取得も `[]` でのインデックス番号の指定もせずに、より可読性の高いプログラムを書くことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="EDqeGWucv9dR" outputId="261c1319-2e24-474f-f283-0d3352e24a27"
# リストをイテラブルオブジェクトに指定
for name in names:
    print('{}さん'.format(name))

# + [markdown] colab_type="text" id="iWEhrw_9v9dS"
# 最初のケースと比べていかがでしょうか。
# 動作としては変わりがありませんが、可読性という観点も重要です。

# + [markdown] colab_type="text" id="eeTHJ3hzPW7U"
# リストをイテラブルオブジェクトとして指定した場合、要素番号を取得できませんが、状況によっては要素番号を使用したいことがあります。
#
# そのような場合は、`enumerate()` という組み込み関数を使います。
# これにイテラブルオブジェクトを渡すと、`(要素番号, 要素)` というタプルを 1 つずつ返すイテラブルオブジェクトになります。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="8dT-qUhGv9dT" outputId="6cdbe801-9f59-49c4-96e6-6d3c54da2e04"
for i, name in enumerate(names):
    message = '{}番目: {}さん'.format(i, name)
    print(message)

# + [markdown] colab_type="text" id="wGwCa9wmP2r4"
# `enumerate()` と同様、`for` 文と合わせてよく使う組み込み関数に `zip()` があります。
#
# `zip()` は、複数のイテラブルオブジェクトを受け取り、その要素のペアを順番に返すイテラブルオブジェクトを作ります。
# このイテラブルオブジェクトは、渡されたイテラブルオブジェクトそれぞれの先頭の要素から順番に、タプルに束ねて返します。
# このイテラブルオブジェクトの長さは、渡されたイテラブルオブジェクトのうち最も短い長さと一致します。

# + colab={"base_uri": "https://localhost:8080/", "height": 53} colab_type="code" id="5w8huhHcQR8F" outputId="528f66f9-c589-481b-f82e-4bf4dfb9982d"
names = ['Python', 'Chainer']
versions = ['3.7', '5.3.0']
suffixes = ['!!', '!!', '?']

for name, version, suffix in zip(names, versions, suffixes):
    print('{} {} {}'.format(name, version, suffix))

# + [markdown] colab_type="text" id="IeQCA00UQYDq"
# `suffixes` の要素数は 3 ですが、より短いイテラブルオブジェクトと共に `zip` に渡されたため、先頭から 2 つ目までしか値が取り出されていません。

# + [markdown] colab_type="text" id="WguFiYgqQnQT"
# ### 条件分岐（if 文）
#
# `if` は、指定した条件が `True` か `False` かによって、処理を変えるための制御構文です。
#
# ![if文](images/02/02_08.png)

# + [markdown] colab_type="text" id="J-Z8g-Klv9dV"
# `elif` と `else` は任意であり、`elif` は 1 つだけでなく複数連ねることができます。
#
# 例えば、0 より大きいことを条件とした処理を書いてみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="E-x_6gXwSsm7" outputId="86d289f1-92f1-4ada-ff3e-017e71f7c4b4"
# if の条件を満たす場合
a = 1

if a > 0:
    print('0より大きいです')
else:
    print('0以下です')

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="BmcHf-w5v9dV" outputId="134ddbe4-5605-4286-945e-a128ad6abc23"
# if の条件を満たさない場合
a = -1

if a > 0:
    print('0より大きいです')
else:
    print('0以下です')

# + [markdown] colab_type="text" id="u0F8tX6FStcQ"
# また、`if` に対する条件以外の条件分岐を追加する場合は、下記のように `elif` を使います。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6li0UyjpStjg" outputId="8e0cecde-fce4-4db6-8d1c-0dbeb80e36b0"
a = 0

if a > 0:    
    print('0より大きいです')
elif a == 0:
    print('０です')
else:
    print('0より小さいです')

# + [markdown] colab_type="text" id="tZ15Hv9Gv9dY"
# ### 繰り返し（while 文）
#
# 繰り返し処理は、`for` 以外にも `while` を用いて記述することもできます。
# `while` 文では、以下のように**ループを継続する条件**を指定します。
# 指定された条件文が `True` である限り、ブロックの部分に記述された処理が繰り返し実行されます。
#
# ![while文](images/02/02_09.png)

# + [markdown] colab_type="text" id="Zjv8d3Rmv9dZ"
# `while` 文を使用した 3 回繰り返すプログラムは下記のとおりです。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="aFhfzaDFv9dZ" outputId="075646ab-8c87-41ea-fd6e-d7ee37334441"
count = 0

while count < 3:
    print(count)
    count += 1

# + [markdown] colab_type="text" id="1Q3QReThv9da"
# ここで使われている `count` という変数は、ループの中身が何回実行されたかを数えるために使われています。
# まず `0` で初期化し、ループ内の処理が一度行われるたびに `count` の値に 1 を足しています。
# この `count` を使った条件式を `while` 文に与えることで、ループを回したい回数を指定しています。
#
# 一方、`while True` と指定すると、`True` は変数ではなく値なので、変更されることはなく、ループは無限に回り続けます。
# `while` 文自体は無限ループの状態にしておき、ループの中で `if` 文を使って、ある条件が満たされた場合はループを中断する、という使い方ができます。
# これには `break` 文が用いられます。
#
# 以下は、`break` 文を使って上のコードと同様に 3 回ループを回すコードです。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="wj65vycFv9da" outputId="14670424-de4d-4493-bf64-2339416c69db"
count = 0

while True:
    print(count)
    count += 1
    
    if count == 3:
        break

# + [markdown] colab_type="text" id="AQFnBfgEv9da"
# `count` の値が 3 と等しいかどうかが毎回チェックされ、等しくなっていれば `break` 文が実行されて `while` ループが終了します。

# + [markdown] colab_type="text" id="2rtEn2APv9db"
# `while` 文を使って、指定された条件を満たして**いない**間ループを繰り返すという処理も書くことができます。`while` 文自体の使い方は同じですが、条件を反転して与えることで、与えた条件が `False` である間繰り返されるようにすることができます。
#
# これには、ブール値を反転する `not` を用います。
# `not True` は `False` を返し、`not False` は `True` を返します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="0PucNNA0v9db" outputId="dc0151bc-a617-48c1-c63b-5826c8ed201c"
not True

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="9iX7JkEOv9db" outputId="3d72981a-bde9-46e1-de00-4fad8b237fda"
not False

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TkSUZ1iZv9dc" outputId="5577f165-0fa0-4ffa-af45-ac847a311049"
not 1 == 2

# + [markdown] colab_type="text" id="0aM2yZ7qv9dd"
# このように、`not` はあとに続くブール値を反転します。
# これを用いて、`count` が 3 **ではない**限りループを繰り返すというコードを `while` 文を使って書いてみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="vhvBNYCXv9dd" outputId="c7651cd7-4f86-47ad-b8a6-c505dd996de5"
count = 0

while not count == 3:
    print(count)
    count += 1


# + [markdown] colab_type="text" id="e96zsBv7IfDn"
# ## 関数
#
# 何かひとまとまりの処理を書いた際には、その処理のためのコードをまとめて、プログラム全体の色々な箇所から再利用できるようにしておくと、便利な場合があります。
# ここでは、処理をひとまとめにする方法の一つとして**関数 (function)** を定義する方法を紹介します。

# + [markdown] colab_type="text" id="RCB98_8xv9df"
# ### 関数を定義する
#
# ![関数の定義](images/02/02_10.png)
#
# 例えば、**受け取った値を 2 倍して表示する関数**を作ってみましょう。
#
# 関数を定義するには、まず名前を決める必要があります。
# 今回は `double()` という名前の関数を定義してみます。
#
# 関数も制御構文と同じく**ヘッダー**と**ブロック**を持っています。

# + colab={} colab_type="code" id="iDh3Mpi_IfDp"
# 関数 double() の定義
def double(x):
    print(2 * x)


# + [markdown] colab_type="text" id="yG-qN405v9dg"
# **関数は定義されただけでは実行されません。**
# 定義した関数を使用するためには、定義を行うコードとは別に、実行を行うコードが必要です。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6QUecRiUv9dg" outputId="7241aa8c-2a9c-4c95-8d63-ebe27a2a6fd8"
# 関数の実行
double(3)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="dR7DqN2fv9dh" outputId="89bb6598-567d-4247-8e0e-2d15072322df"
double(5)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="wjStuxcBv9di" outputId="138a4607-ebf9-48f3-e4f5-96863b9eec0f"
double(1.5)


# + [markdown] colab_type="text" id="CRggp-QcIfDs"
# `double(x)` における `x` のように、関数に渡される変数や値のことを**引数 (argument)** といいます。
# 上の例は、名前が `double` で、1つの引数 `x` をとり、`2 * x` という計算を行い、その結果を表示しています。

# + [markdown] colab_type="text" id="VDqeEdX5v9dj"
# ### 複数の引数をとる関数
#
# 複数の引数をとる関数を定義する場合は、関数名に続く `()` の中に、カンマ `,` 区切りで引数名を並べます。
#
# 例えば、引数を 2 つとり、足し算を行う関数 `add()` を作ってみましょう。

# + colab={} colab_type="code" id="ARE2LU8Qv9dl"
# 関数の定義
def add(a, b):
    print(a + b)


# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Zte-_M1-v9dl" outputId="cf78e6e7-522c-4551-b876-8f7f4f582d92"
# 関数の実行
add(1, 2)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="7HUAVoGhv9dm" outputId="8e0d0ec9-9774-42b6-ef79-52b20cf88b8a"
add(3, 2.5)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="IjMBFjvmv9dm" outputId="75da673b-d203-460c-f054-205cbe9ace97"
add(1, -5)


# + [markdown] colab_type="text" id="mwZ-T1CTv9dn"
# 今回の `double()` や `add()` は定義を行い自作した関数ですが、Python には予め多くの関数が定義されています。
# そのような関数を**組み込み関数 (built-in function)** と呼びます。
# すでに使用している `print()` や `len()`, `range()` などが、これに該当します。
# 組み込み関数の一覧は[こちら](https://docs.python.org/ja/3/library/functions.html)で確認することができます。
#

# + [markdown] colab_type="text" id="A8XpuE_av9dn"
# ### 引数をとらない関数
#
# 引数をとらない関数を定義する場合でも、関数名の後に `()` を加える必要があります。
#
# 例えば、実行するとメッセージを表示する関数を定義して、実行してみましょう。

# + colab={} colab_type="code" id="z2Xej00Bv9dn"
# 引数のない関数の定義
def hello():
    print('Chainerチュートリアルにようこそ')


# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="peShhHg2v9dn" outputId="b437e0f9-1c01-4d93-a13c-2d50cf8202f9"
# 引数のない関数の実行
hello()


# + [markdown] colab_type="text" id="BZGSKTbcv9do"
# ### 引数のデフォルト値
#
# 引数には、あらかじめ値を与えておくことができます。
# これは、引数をとる関数を定義する際に、何も引数に値が渡されなかったときにどのような値がその引数に渡されたことにするかをあらかじめ決めておける機能で、その値のことを**デフォルト値**と呼びます。
#
# 例えば、上の `hello()` という関数に、`message` という引数をもたせ、そこにデフォルト値を設定しておきます。

# + colab={} colab_type="code" id="uw2d8qDov9do"
def hello(message='Chainerチュートリアルにようこそ'):
    print(message)


# + [markdown] colab_type="text" id="7i8TQcfzv9do"
# この関数は引数に何も与えずに呼び出すと、「Chainerチュートリアルにようこそ」というメッセージを表示し、引数に別な値が渡されると、その値を表示します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Fh4VEHmMv9dp" outputId="d284fd9b-4a96-4039-f585-a01804e360e9"
hello()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="p2g-ULOWv9dp" outputId="9dd78b5c-4185-4f40-a40f-9d1bdfe3309a"
hello('Welcome to Chainer tutorial')


# + [markdown] colab_type="text" id="GWhAbt3ov9dq"
# デフォルト値が与えられていない引数は、関数呼び出しの際に必ず何らかの値が渡される必要がありますが、デフォルト値を持つ場合は、何も指定しなくても関数を呼び出すことができるようになります。

# + [markdown] colab_type="text" id="DpwRYy4ov9dq"
# ### 返り値のある関数
#
# 上で定義した足し算を行う関数 `add()` では、計算結果を表示するだけで、計算結果を呼び出し元に戻していませんでした。
# そのため、このままでは計算結果を関数の外から利用することができません。
#
# そこで、`add()` 関数の末尾に `return` 文を追加して、計算結果を呼び出し元に返すように変更してみましょう。

# + colab={} colab_type="code" id="ZCtzpxG1v9dq"
# 返り値のある関数の定義
def add(a, b):
    return a + b


# + [markdown] colab_type="text" id="M9c1_e-sv9dr"
# このように、呼び出し元に返したい値を `return` に続いて書くと、その値が `add()` 関数を呼び出したところへ戻されます。
# `return` で返される値のことを**返り値 (return value)** と言います。
#
# 以下に、計算結果を `result` という変数に格納し、表示する例を示します。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="RPJnjuL5v9dr" outputId="48546d3e-a444-4109-ddf0-7df7b14b9236"
result = add(1, 3)

result

# + [markdown] colab_type="text" id="tagr61vUv9ds"
# 計算結果が `result` に格納されているので、この結果を用いてさらに別の処理を行うことができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Qe8Enb6Ev9ds" outputId="99526b92-37c3-4832-caef-f17aa45701d9"
result = add(1, 3)

result_doubled = result * 2

result_doubled

# + [markdown] colab_type="text" id="fP4OdcI-v9ds"
# また、返り値は「呼び出し元」に返されると書きました。
# この「呼び出し元」というのは、関数を呼び出す部分のことで、上のコードは `add(1, 3)` の部分が `4` という結果の値になり、それが左辺の `result` に代入されています。
#
# これを用いると、例えば「2 と 3 を足した結果と、1 と 3 を足した結果を、掛け合わせる」という計算が、以下のように書けます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="gs0hBjg6v9ds" outputId="4c76fe15-686d-4aed-f87c-ec55d177728c"
add(2, 3) * add(1, 3)

# + [markdown] colab_type="text" id="BJnkcc_Qv9dt"
# ### 変数のスコープ
#
# 関数の中で定義した変数は基本的には関数の外では利用できません。
# 例えば、以下の例を見てみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="L8vtxEV7v9dt" outputId="509fbb4d-0b47-4f37-d38c-51018e6f19b2"
a = 1

# 関数の内部で a に 2 を代入
def change():
    a = 2
    
change()

a

# + [markdown] colab_type="text" id="Z9slKEKXv9du"
# 関数の外で `a = 1` と初期化した変数と同じ名前の変数に対して、`change()` 関数の内部で `a = 2` という代入を行っているにもかかわらず、`change()` 関数の実行後にも関数の外側では `a` の値は 1 のままになっています。
# **関数の外側で定義された変数** `a` **に、関数内部での処理が影響していないことがわかります。**
#
# なぜこうなるかというと、関数の中で変数に値が代入されるとき、その変数はその関数の**スコープ (scope)** でだけ有効な**ローカル変数**になり、関数の外にある同じ名前の変数とは別のものを指すようになるためです。
# スコープとは、その変数が参照可能な範囲のことです。
# 上の例では、`a = 2` の代入を行った時点で`change()` 関数のスコープに `a` という変数が作られ、`change()` 関数の中からは `a` といえばこれを指すようになります。関数から抜けると、`a` は 1 を値に持つ外側の変数を指すようになります。
#
# ただし、代入を行わずに、参照するだけであれば、関数の内側から外側で定義された変数を利用することができます。

# + colab={"base_uri": "https://localhost:8080/", "height": 53} colab_type="code" id="XgGcEUSQv9du" outputId="0871b4ef-d71a-4f3f-c556-fc6429e0ee94"
a = 1

def change():
    print('From inside:', a)
    
change()

print('From outside:', a)

# + [markdown] colab_type="text" id="vAHe5xDGv9dv"
# この場合は、`change()` 関数のスコープには `a` という変数は作られないので、関数の中で `a` といえば外側で定義された変数を指します。
#
# 関数の外で定義された変数は**グローバル変数**と呼ばれます。
# グローバル変数は、特に特別な記述を要せず参照することはできますが、関数の中で**代入**を行う場合は、`global` 文を使って、代入先をグローバル変数とする宣言を行う必要があります。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="D1uNClamv9dv" outputId="d44e9ee4-8250-4236-9476-0cc610cb9572"
a = 1

def change():
    global a  # a をグローバル変数である宣言
    a = 2       # グローバル変数への代入

# 関数の実行
change()

# 結果の確認 <- a の値が上書きされている
a


# + [markdown] colab_type="text" id="MSTyBWulv9dv"
# `global a` という行を `change()` 関数内で `a` という変数を使用する前に追加すると、その行以降は `a` という変数への代入も関数の外側で定義されたグローバル変数の `a` に対して行われます。

# + [markdown] colab_type="text" id="FBe0bEnnVoE5"
# ## クラス
#
# **オブジェクト指向プログラミング (object-oriented programming)** の特徴の一つである**クラス (class)** は、**オブジェクト (object)** を生成するための設計図にあたるものです。
# まず、クラスとは何か、オブジェクトとは何かについて説明します。
#
# ここで、唐突に感じられるかもしれませんが、家を何軒も建てるときのことを考えましょう。
# それぞれの家の形や大きさ、構造は同じですが、表札に書かれている名前は異なっているとします。
# この場合、家の設計図は同じですが、表札に何と書くか、において多少の変更がそれぞれの家ごとに必要となります。
# この**全ての家に共通した設計図の役割を果たすのがクラス**です。
# そして、設計図は、家として現実に存在しているわけではありませんが、個別の家は、現実に家としての**実体**を持って存在しています。
# よって、**設計図に基づいて個別の家を建てる**ということを抽象的に言うと、**クラスから実体を作成する**、となります。
# クラスから作成された実体のことを**インスタンス (instance)** または**オブジェクト (object)** とも呼び、**クラスから実体を作成する**という操作のことを**インスタンス化 (instantiation)** と呼びます。

# + [markdown] colab_type="text" id="M1XEvf6Kv9dw"
# ### クラスの定義
#
# それでは、家の設計図を表す `House` というクラスを定義してみましょう。
# `House` クラスには、インスタンス化されたあとに、各インスタンス、すなわち誰か特定の人の家ごとに異なる値を持つ、`name_plate` という変数を持たせてみます。
#
# `name_plate` という変数には、個別の家の表札に表示するための文字列が与えられますが、クラスを定義する際には「`name_plate` という変数を持つことができる」ようにしておくだけでよく、**実際にその変数に何か具体的な値を与える必要はありません。**
# クラスは、**設計図**であればよく、具体的な値を持たせなくてもよいためです。
# 具体的な値は、個別の家を作成するとき、すなわちインスタンス化の際に与え、各インスタンスが `name_plate` という値に自分の家の表札の名前を保持するようにします。
#
# このような、インスタンスに属している変数を**属性 (attribute)** と呼びます。同様に、インスタンスから呼び出すことができる関数のことを**メソッド (method)** と呼びます。
#
# クラスは、以下のような構文を使って定義します。
#
# ![クラス](images/02/02_11.png)
#
# 具体的には、以下のようになります。

# + colab={} colab_type="code" id="9ZrUz4kHv9dw"
# クラスの定義
class House:

    # __init__() メソッドの定義
    def __init__(self, name):
        self.name_plate = name


# + [markdown] colab_type="text" id="151zWM4Dv9dw"
# ここで、`__init__()` という名前のメソッドが `House` クラスの中に定義されています。
# メソッドの名前は自由に名付けることができますが、いくつか特別な意味を持つメソッド名が予め決められています。
# `__init__()` はそういったメソッドの一つで、**インスタンス化する際に自動的に呼ばれるメソッド**です。
#
# `House` クラスの `__init__()` は、`name` という引数をとり、これを `self.name_plate` という変数に代入しています。
# この `self` というのは、クラスがインスタンス化されたあと、作成されたインスタンス自身を参照するのに用いられます。
# これを使って、`self.name_plate = name` とすることで、作成された個別のインスタンスに属する変数 `self.name_plate` へ、引数に渡された `name` が持つ値を代入することができます。
# `self` が指すものは、各インスタンスから見た「自分自身」なので、各インスタンスごとに異なります。
# これによって、`self.name_plate` は各インスタンスに紐付いた別々の値を持つものとなります。
#
# メソッドは、インスタンスから呼び出されるとき自動的に第一引数にそのインスタンスへの参照を渡します。
# そのため、メソッドの第一引数は `self` とし、渡されてくる自分自身への参照を受け取るようにしています。
# ただし、呼び出す際には**そのインスタンスを引数に指定する必要はありません。**
# 以下に具体例を示し、再度このことを確認します。
#
# それでは、上で定義した `House` クラスのインスタンスを作成してみます。
# クラスのインスタンス化には、クラス名のあとに `()` を追加して、クラスを呼び出すような記法を使います。
# この際、関数を呼び出すときと同様にして、`()` に引数を渡すことができます。
# その引数は、`__init__()` メソッドに渡されます。

# + colab={} colab_type="code" id="PE3dT2ytv9dx"
my_house = House('Chainer')


# + [markdown] colab_type="text" id="9xHHa6rwv9dx"
# `House` というクラスの `__init__()` メソッドに、`'Chainer'` という文字列を渡しています。
# `my_house` が、`House` クラスから作成されたインスタンスです。
# ここで、クラス定義では `__init__()` メソッドは `self` と `name` という 2 つの引数をとっていましたが、呼び出しの際には `'Chainer'` という一つの引数しか与えていませんでした。
# この `'Chainer'` という文字列は、1 つ目の引数であるにも関わらず、`__init__()` メソッドの定義では 2 つ目の引数であった `name` に渡されます。
# 前述のように、**メソッドは、インスタンスから呼び出されるとき自動的に第一引数にそのインスタンスへの参照を渡す**ためです。
# この自動的に渡される自身への参照は、呼び出しの際には明示的に指定しません。
# また、かならず 1 つ目の引数に自動的に渡されるため、呼び出し時に明示的に与えられた引数は 2 つ目以降の引数に渡されたものとして取り扱われます。
#
# それでは次に、このクラスに `hello()` というメソッドを追加し、呼び出すと誰の家であるかを表示するという機能を実装してみます。

# + colab={} colab_type="code" id="k2Zcvs1xv9dx"
# クラスの定義
class House:

    # __init__() の定義
    def __init__(self, name):
        self.name_plate = name

    # メソッドの定義
    def hello(self):
        print('{}の家です。'.format(self.name_plate))


# + [markdown] colab_type="text" id="oTQagqYdv9dz"
# それでは、2 つのインスタンスを作成して、それぞれから `hello()` メソッドを呼び出してみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 53} colab_type="code" id="MP8iXQM0v9dz" outputId="74495bc8-073e-4d46-addf-ee02f45ed8b8"
sato = House('佐藤')
suzuki = House('スズキ')

sato.hello()   # 実行の際には hello() の引数にある self は無視
suzuki.hello() # 実行の際には hello() の引数にある self は無視


# + [markdown] colab_type="text" id="_vI_Dqjdv9d0"
# `sato` というインスタンスの `name_plate` 属性には、`'佐藤'` という文字列が格納されています。  
# `suzuki` というインスタンスの `name_plate` 属性には、`'スズキ'` という文字列が格納されています。  
# それぞれのインスタンスから呼び出された `hello()` メソッドは、`self.name_plate` に格納された別々の値を `print()` を用いて表示しています。
#
# このように、同じ機能を持つが、インスタンスによって保持するデータが異なったり、一部の動作が異なったりするようなケースを扱うのにクラスを利用します。
# Python の `int` 型、`float` 型、`str` 型…などは、実際には `int` クラス、`float` クラス、`str` クラスであり、それらの中では個別の変数（インスタンス）がどのような値になるかには関係なく、同じ型であれば共通して持っている機能が定義されています。
# `5` や `0.3` や `'Chainer'` などは、それぞれ `int` クラスのインスタンス、`float` クラスのインスタンス、`str` クラスのインスタンスです。
#
# 以上から、クラスを定義するというのは、**新しい型を作る**ということでもあると分かります。

# + [markdown] colab_type="text" id="0M1dmJSacA7g"
# ### 継承
#
# あるクラスを定義したら、その一部の機能を変更したり、新しい機能を付け足したりしたくなることがあります。
# これを実現する機能が**継承 (inheritance)** です。
# 例えば、`Link` というクラスを定義し、そのクラスを継承した `Chain` という新しいクラスを作ってみましょう。
# まず、`Link` クラスを定義します。

# + colab={} colab_type="code" id="wn9XJxjpv9d0"
class Link:

    def __init__(self):
        self.a = 1
        self.b = 2


# + [markdown] colab_type="text" id="Qn0fSwvyv9d1"
# この `Link` というクラスは、インスタンス化を行う際には 1 つも引数をとりませんが、属性として `a` と `b` の 2 つの変数を保持し、それぞれには `__init__()` メソッドで 1 と 2 という値が代入されます。
# このクラスのインスタンスを作成してみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="BsfYK88qv9d1" outputId="fe561266-04ce-4e67-9c08-79027499b92f"
l = Link()

l.a

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="JA6sXOrvv9d2" outputId="e58234d5-268a-4668-e946-e9fe18ef7c23"
l.b


# + [markdown] colab_type="text" id="DxipToKmv9d2"
# `l` という `Link` クラスのインスタンスが持つ 2 つの属性を表示しています。
# インスタンス化を行った際に `__init__()` メソッドの中で代入していた値が、表示されています。
#
# 次に、このクラスを**継承**する、`Chain` というクラスを定義してみます。
# 継承を行う場合は、クラス定義の際にクラス名に続けて `()` を書き、その中にベースにしたいクラスの名前を書きます。
# `()` の中に書かれたクラスのことを、定義されるクラスの**親クラス**といいます。
# それに対し、`()` の中に書かれたクラスからみると、定義されるクラスは**子クラス**と呼ばれます。
# 親から子へ機能が受け継がれるためです。

# + colab={} colab_type="code" id="A4eTa6jZv9d3"
class Chain(Link):
    
    def sum(self):
        return self.a + self.b


# + [markdown] colab_type="text" id="dhxR4J0mv9d3"
# `Chain` クラスは `__init__()` メソッドの定義を持ちません。
# `__init__()` メソッドが定義されていない場合、親クラスの `__init__()`  メソッドが自動的に呼び出されます。
# そのため、`Chain` クラスでは一見何も属性を定義していないように見えますが、インスタンス化を行うと親クラスである `Link` の `__init__()`  メソッドが自動的に実行され、`a`、`b` という属性が定義されます。
# 以下のコードで確認してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="ipmqPwsKv9d3" outputId="2c340a24-3414-4c84-da5f-48f911366135"
# Chain クラスをインスタンス化
c = Chain()

c.a

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="lWL0VTMOv9d4" outputId="0342c542-8a1a-4793-f54e-496f31a8a5ba"
c.b

# + [markdown] colab_type="text" id="nVjBNK1Cv9d4"
# `Chain` クラスの `sum()` メソッドでは、この親クラスの `__init__()`  メソッドで定義されている 2 つの属性を足し合わせて返しています。
# 今作成したインスタンスから、この `sum()` メソッドを呼び出してみます。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="pEf10A44v9d5" outputId="18759ed5-2255-473c-d7c5-ab922b8d1491"
# sum メソッドを実行
c.sum()


# + [markdown] colab_type="text" id="QI1v_Ib0v9d5"
# このように、**親クラスを継承し、親クラスに無かった新しい機能が追加された、新しいクラスを定義することができます。**
#
# それでは、この `Chain` というクラスにも `__init__()`  メソッドを定義して、新しい属性 `c` を定義し、`sum()` メソッドでは親クラスの `a`、`b` という属性とこの新たな `c` という属性の 3 つの和を返すように変更してみます。

# + colab={} colab_type="code" id="ds9A6Xfpv9d5"
class Chain(Link):

    def __init__(self):
        self.c = 5  # self.c を新たに追加
    
    def sum(self):
        return self.a + self.b + self.c

# インスタンス化
C = Chain()

# + colab={"base_uri": "https://localhost:8080/", "height": 296} colab_type="code" id="_2_7avSHv9d6" outputId="c80707d3-ae9d-4b24-a0c9-ee6c492ed6c7"
# error
C.sum()


# + [markdown] colab_type="text" id="ZfTe1NMUv9d6"
# エラーが出ました。
#
# **エラーメッセージを読みましょう。**
#
# > AttributeError: 'Chain' object has no attribute 'a'
#
# `'Chain'` というオブジェクトは、`'a'` という名前の属性を持っていない、と言われています。
# `a` という属性は、`Chain` の親クラスである `Link` の `__init__()`  メソッドで定義されています。
# そのため、`Chain` クラスをインスタンス化する際に、親クラスである `Link` の `__init__()`  メソッドが呼ばれているのであれば、このエラーは起こらないはずです。
# なぜエラーとなってしまったのでしょうか。
#
# それは、`Chain` クラスにも `__init__()` メソッドを定義したため、親クラスである `Link` の `__init__()`  メソッドが上書きされてしまい、実行されなかったためです。
# しかし、親クラスの `__init__()`  メソッドを明示的に呼ぶことで、これは解決できます。
#
# それには、`super()` という組み込み関数を用います。
# これを用いると、子クラスから親クラスを参照することができます。

# + colab={} colab_type="code" id="_NcT6aN7v9d6"
class Chain(Link):

    def __init__(self):
        # 親クラスの `__init__()` メソッドを呼び出す
        super().__init__()
        
        # self.c を新たに追加
        self.c = 5
    
    def sum(self):
        return self.a + self.b + self.c

# インスタンス化
c = Chain()

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="FW__tTtGv9d7" outputId="4fab420e-d2af-4d1f-9c3a-f3e5a1fe5820"
c.sum()


# + [markdown] colab_type="text" id="HcDciFKRv9d7"
# 今回はエラーが起きませんでした。
# `Link` クラスの `__init__()`  メソッドの冒頭で、まず親クラスの `__init__()`  メソッドを実行し、`a`、`b` という属性を定義しているためです。
#
# あるクラスを継承して作られたクラスを、さらに継承して別のクラスを定義することもできます。

# + colab={} colab_type="code" id="RmL3FvD-v9d7"
class MyNetwork(Chain):
    
    def mul(self):
        return self.a * self.b * self.c


# + [markdown] colab_type="text" id="21eb0EOKv9d8"
# `MyNetwork` クラスは、`Link` クラスを継承した `Chain` クラスをさらに継承したクラスで、`a`、`b`、`c` という 3 つの属性を掛け合わせた結果を返す `mul()` というメソッドを持ちます。
#
# このクラスのインスタンスを作成し、`mul()` を実行してみましょう。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="Cq4lJIm3v9d8" outputId="afcc0ba0-f917-41ca-d06a-64b42e1e2456"
net = MyNetwork()

net.mul()

# + [markdown] colab_type="text" id="VujV1nqfv9d8"
# $1 \times 2 \times 5 = 10$ が返ってきました。

# + [markdown] colab_type="text" id="mIPsXEy0v9d9"
# 以上で、Python の基本についての解説を終了します。
# Python には他にもここでは紹介されていない多くの特徴や機能があります。
# さらに詳しく学びたい方は、[Pythonチュートリアル](https://docs.python.org/ja/3/tutorial/index.html) などを参照してください。
