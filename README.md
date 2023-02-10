# PG-DSコース課題開発リポジトリ
PlayGround/データサイエンスコース で用いるための課題を改善、開発していくリポジトリです。  

このリポジトリの課題の雛形は[Introduction to Deep Learning: Chainer Tutorials](https://tutorials.chainer.org/)です。  

## LICENSE
オリジナルのChainerTutorialのライセンスは ”BSD 3-Clause”です。  
[ライセンス原文](LICENSE)  
[ライセンス日本語訳](LICENSE-ja)

## 環境
とりあえず、各自使用するライブラリをインストールする運用です  
### pre-commitのセットアップについて
このリポジトリでは`pre-commit`を使うため、各自設定が必要です。  
1. pre-commitのインストール   
    `pip install pre-commit`  
2. pre-commitの設定  
    `pre-commit install`  
これにより`.git/hooks/pre-commit`に`pre-commit`がインストールされ、以後`pre-commit`が実行されるようになります。  

## ディレクトリ構造
- ja/
    元の`ChainerTutorial`で使用されていたnotebookが保存されています。  
    ここにあるファイルは編集しないでください。

- ja-edited/
    ここにあるnotebookを編集して作業します。  
    
- scripts/ja-edited/
    ja-edited以下の`.ipynb`形式のファイルが`.py`に変換されてここに保存されます。  
    変換には[`pre-commit`](https://pre-commit.com/)と[`jupytext`](https://jupytext.readthedocs.io/en/latest/index.html)が使用されています。

## 開発ルール
実装は全て`/ja-edited/*.ipynb`に対して行ってください。
### ポイント
- 新規課題の追加・課題の変更を行う際は必ずissueを立てる
- 課題の追加は `feature/追加課題を表すタイトル`ブランチで行う
- 課題の変更は `fix/変更を表すタイトル`ブランチで行う
- 変更を追加し終わったらpushしてPRを送信し、レビューをもらう
    - レビュワー：(wip)

### pre-commit
このプロジェクトでは`pre-commit`を使用して`ipynb`ファイルと`py`ファイルの内容を同期させています。  
詳細は[.pre-commit-config.yaml]()を参照してください。
#### jupytext
pre-commitによってjupytextコマンドが実行され、
`ja-edited/hoge.ipynb`の内容が`scripts/ja-edited/hoge.py`に反映されます。  
**この際上記のファイルの両方がステージングされている必要があります。**
