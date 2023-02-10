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

# + [markdown] colab_type="text" id="_CMW13uMBlpk"
# # はじめに
#
#
# このチュートリアルは、機械学習やディープラーニングの仕組みや使い方を理解したい**大学学部生**以上の方に向けて書かれたオンライン学習資料です
#
# 機械学習の勉強を進めるために必要な数学の知識から、Python というプログラミング言語を用いたコーディングの基本、機械学習・ディープラーニングの基礎的な理論、画像認識や自然言語処理などに機械学習を応用する方法に至るまで、幅広いトピックを解説しています。
#
# 機械学習を学び始めようとすると、ある程度、線形代数や確率統計といった数学の知識から、何らかのプログラミング言語が使えることなどが必要となってきます。
# しかし、そういった数学やプログラミングの全てに精通していなければ機械学習について学び始められないかというと、必ずしもそうではありません。
#
# 本チュートリアルでは、機械学習やディープラーニングに興味を持った方が、まず必要になる最低限の数学とプログラミングの知識から学び始められるように、資料を充実させています。
#
# そのため、できる限りこのサイト以外の教科書や資料を探さなくても、**このサイトだけで機械学習・ディープラーニングに入門できる**ことを目指して、作られています。初学者の方が「何から学び始めればいいのか」と迷うことなく学習を始められることを目指したサイトです。
#
# また、本チュートリアルの特徴として、資料の中に登場するコードが、Google Colaboratory というサービスを利用することで**そのままブラウザ上で実行できるようになっている**という点があります。
#
# ブラウザだけでコードを書き、実行して、結果を確認することができれば、説明に使われたサンプルコードを実行して結果を確かめるために、手元のコンピュータで環境構築を行う必要がなくなります。 
#
# 本章ではまず、この **Google Colaboratory** というサービスの利用方法を説明します。

# + [markdown] colab_type="text" id="TK3cXCQuBlpm"
# ## 必要なもの
#
# - Google アカウント（お持ちでない場合は、こちらからお作りください：[Google アカウントの作成](https://accounts.google.com/signup)）
# - ウェブブラウザ（ Google Colaboratory はほとんどの主要なブラウザで動作します。PC 版の Chrome と Firefox では動作が検証されています。）

# + [markdown] colab_type="text" id="D3QflLv0qdiy"
# ## Google Colaboratory の基本
#
# Google Colaboratory（以下 Colab ）は、クラウド上で [Jupyter Notebook](https://jupyter.org/)  環境を提供する Google のウェブサービスです。Jupyter Notebook はブラウザ上で主に以下のようなことが可能なオープンソースのウェブアプリケーションであり、データ分析の現場や研究、教育などで広く用いられています。
#
# - プログラムを実行と、その結果の確認
# - Markdown と呼ばれる文章を記述するためのマークアップ言語を使った、メモや解説などの記述の追加
#
# Colab では無料で GPU も使用することができますが、そのランタイムは**最大 12 時間**で消えてしまうため、長時間を要する処理などは別途環境を用意する必要があります。
# 学びはじめのうちは、数分から数時間程度で終わる処理がほとんどであるため、気にする必要はありませんが、本格的に使っていく場合は有料のクラウドサービスを利用するなどして、環境を整えるようにしましょう。
#
# 以降では、その基本的な使い方を説明します。

# + [markdown] colab_type="text" id="SNVvaFdwBlpn"
# ### Colab を開く
#
# まずは以下のURLにアクセスして、ブラウザで Colab を開いてください。
#
# [https://colab.research.google.com/](https://colab.research.google.com/)
#
# 「Colaboratory へようこそ」というタイトルの Jupyter Notebook が表示されます。
#
# 次に、タイトルの下にある 「ファイル」 から、「Python 3 の新しいノートブック」 を選択し、まっさらな Jupyter Notebook を作成しましょう。
#
# ![create new notebook](images/01/01_create_new_notebook.png)
#
# Google アカウントにまだログインしていなかった場合は、以下のようなメッセージが表示されます。
#
# ![please login](images/01/01_02.png)
#
# その場合は、「ログイン」 をクリックして、Google アカウントでログインしてください。
#
# ログインが完了すると、以下のような画面が表示され、準備完了です。
# もうすでに Python を使ったプログラミングを開始する準備が整っています。
#
# ![new python3 notebook](images/01/01_03.png)

# + [markdown] colab_type="text" id="SWEJAKWMBlpo"
# ### Open in Colab ボタン
#
# このチュートリアルの一部の章には、`Open in Colab` と書かれた以下のようなボタンがページ上部に設置されています。
#
# [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chainer/tutorials/blob/master/ja/01_Welcome_to_Chainer_Tutorial_ja.ipynb)
#
# このボタンを押すと、ブラウザで見ている資料が、Colab 上で  Jupyter Notebook として開かれます。
# すると、チュートリアルの中で説明に用いられているコードを、**実際に実行して結果を確認することができます。**
#
# それでは、早速上のボタンか、このページの上部に配置されている `Open in Colab` ボタンを押して、このページを Colab で開いてください。  
# すると、`Playground モード` という編集不可な状態でノートブックが Colab 上で開かれます。
# そこで、下図の位置にある `ドライブにコピー` というボタンを押して、自分の Google Drive 上にこのノートブックをコピーしてください。
# このボタンを押すと、コピーされたノートブックが自動的に開き、以降は内容に編集を加えたり、コードを実行したりすることができます。
#
# ![copy to mydrive](images/01/01_04.png)
#
# この
#
# 1. `Open in Colab` から Colab へ移動
# 2. 自分のドライブへノートブックをコピーする
# 3. コードを実行しながら解説を読んでいく
#
# という手順が、本チュートリアルサイトのおすすめの利用方法です。

# + [markdown] colab_type="text" id="k4ak2UP9Blpp"
# ## Colab の基本的な使い方
#
# Colab 上の Jupyter Notebook を以降、単に**ノートブック**と呼びます。  
#
# ノートブックは、複数の**セル**と呼ばれるブロックを持つことができます。
# 新しいノートブックを作った直後では、何も書かれていないセルが一つだけ存在している状態になっています。
# セルの内側のどこかをクリックすると、そのセルを選択することができます。
#
# セルには、**コードセル**と**テキストセル**の 2 種類があります。
# **コードセル** は Python のコードを書き込み、実行するためのセルであり、**テキストセル**は、Markdown 形式で文章を書くためのセルです。
#
# それぞれのセルタイプについてもう少し詳しく説明をします。

# + [markdown] colab_type="text" id="9hYkVljaBlpq"
# ### コードセル
#
# コードセルは、Python のコードを書き込み、実行することができるセルです。
# 実行するには、コードセルを選択した状態で、`Ctrl + Enter` または `Shift + Enter` を押します。
# 試しに、下のセルを選択して、`Ctrl + Enter` を押してみてください。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="EaOJalpbBlpr" outputId="40291477-aa22-4151-da78-d2ae6d2eb627"
print('Hello world!')

# + [markdown] colab_type="text" id="8QsnHDylXQrb"
# すぐ下に、Hello world! という文字列が表示されました。
# 上のセルに書き込まれているのは Python のコードで、与えられた文字列を表示する関数である `print()` に、`'Hello world!'` という文字列を渡しています。
# これを今実行したため、その結果が下に表示されています。
#
# プログラミング言語の Python については、[次の章](https://tutorials.chainer.org/ja/02_Basics_of_Python.html) でより詳しく解説します。

# + [markdown] colab_type="text" id="f7vtQ2SmBlpx"
# ### テキストセル
#
# テキストセルでは、Markdown 形式で記述された文章を扱います。
# 試しに、このセルを**ダブルクリック**してみてください。
# テキストセルが編集モードになり、Markdown 形式で文章を装飾するための、先程までは表示されていなかった記号が見えるようになります。
#
# その状態で `Shift + Enter` を押してみましょう。  
#
# もとのレンダリングされた文章の表示に戻ります。

# + [markdown] colab_type="text" id="wEwqOW9bBlpy"
# ### Colab から Google Drive を使う
#
# Google Drive というオンラインストレージサービスを Colab で開いたノートブックから利用することができます。
# ノートブック中でコードを実行して作成したファイルなどを保存したり、逆に Google Drive 上に保存されているデータを読み込んだりすることができます。
#
# Colab 上のノートブックから Google Drive を使うには、Colab 専用のツールを使って、`/content/drive` というパスに現在ログイン中の Google アカウントが持っている Google Drive のスペースをマウントします。

# + colab={} colab_type="code" id="TI3-V_gN3Ekr"
from google.colab import drive
drive.mount('/content/drive')

# + [markdown] colab_type="text" id="zllU5vanBlp2"
# このノートブックを Colab で開いてから初めて上のコードセルを実行した場合は、以下のようなメッセージが表示されます。
#
# ![please authorize](images/01/01_05.png)
#
# 指示に従って表示されているURLへアクセスしてください。
# すると、「アカウントの選択」と書かれたページに飛び、すでにログイン済みの場合はログイン中の Google アカウントのアイコンやメールアドレスが表示されています。
# 利用したいアカウントをクリックして、次に進んで下さい。
# すると次に、`Google Drive File Stream が Google アカウントへのアクセスをリクエストしています` と書かれたページに飛びます。
#
# ![access request](images/01/01_06.png)
#
# 右下に「許可」と書かれたボタンが見えます。
# こちらをクリックしてください。
# すると以下のように認証コードが記載されたページへ移動します。
#
# ![access code](images/01/01_07.png)
#
# （この画像では認証コード部分をぼかしています）
# このコードを選択してコピーするか、右側にあるアイコンをクリックしてコピーしてください。
#
# 元のノートブックへ戻り、`Enter your authorization code:` というメッセージの下にある空欄に、先程コピーした認証コードを貼り付けて、Enter キーを押してください。
#
# **Mounted at /content/drive** と表示されたら、準備は完了です。
#
# 以下のセルを実行して、自分の Google Drive が Colab からアクセス可能になっていることを確認してください。

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="osShvuIQ3GFy" outputId="e244c570-af51-4af8-a9f1-5a68da0fa41b"
# 'My Drive'の表記が出ていればマウントがうまく行われています。
# !ls 'drive/'

# + [markdown] colab_type="text" id="DbvFPwpova8M"
# 上のセルで実行しているのは Python のコードではありません。
# Jupyter Notebook では、コードセル中で `!` が先頭に付いている行は特別に解釈されます。`!ls` は、次に続くディレクトリの中にあるファイルまたはディレクトリの一覧を表示せよ、という意味です（[注釈1](#note1)）。

# + [markdown] colab_type="text" id="jZNTuBQ54BSu"
# ### Colab の便利なショートカット
#
# Colab を使用中に、セルのタイプの変更やセルの複製・追加などの操作をする場合は、メニューから該当する項目を選ぶ方法以外に、キーボードショートカットを利用する方法もあります。
#
# 下記によく使う**ショートカットキー**をまとめておきます。
# 多くのショートカットキーは**二段階**になっており、まず `Ctrl + M` を押してから、それぞれの機能によって異なるコマンドを入力する形になっています。
#
# | 説明                 | コマンド      |
# | -------------------- | ------------- |
# | Markdownモードへ変更 | Ctrl + M → M |
# | Codeモードへ変更     | Ctrl + M → Y |
# | セルの実行           | Shift + Enter |
# | セルを上に追加       | Ctrl + M → A |
# | セルを下に追加       | Ctrl + M → B |
# | セルのコピー         | Ctrl + M → C |
# | セルの貼り付け       | Ctrl + M → V |
# | セルの消去           | Ctrl + M → D |
# | コメントアウト       | Ctrl + /      |
#
# コメントアウトとは、コード中で実行時に無視したい行やコメントを選択した状態で行う操作です。
# Python では、`#` の後に続く文字列は全て、コメントとして無視され、実行時に評価されることはありません。

# + [markdown] colab_type="text" id="44vOyaBKEk3m"
# ###  GPU を使用する
#
# Colab では GPU を無料で使用することができます。
# 初期設定では GPU を使用しない設定となっているため、GPU を使用する場合は設定を変更する必要があります。
#
# GPU を使用する場合は、画面上部のタブの中の 「Runtime」 (または「ランタイム」) をクリックし、「Change runtime type」 (または「ランタイムのタイプを変更」)を選択します。  
#
# そして、下記の画像の様に 「Hardware accelerator」  (または「ハードウェアアクセラレータ」)を GPU に変更します。  
#
# ![GPUの設定](images/01/01_08.png)
#
# これで Colab 上で GPU を使用できるようになりました。
#
#

# + [markdown] colab_type="text" id="QoQHVO6rva8O"
# これで、チュートリアルの本編に入っていく準備が完了しました。次の章では、Python というプログラミング言語の基本について解説します。

# + [markdown] colab_type="text" id="rot1jrxLy47Y"
# <hr />
# <div class="alert alert-info">
# **注釈 1**
#
# `ls` はシェルコマンドの 1 つです。
#  
# [▲上へ戻る](#ref_note1)
# </div>
#
