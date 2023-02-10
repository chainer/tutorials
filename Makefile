sync:
	 find scripts/ja-edited/*.py| sed -e "s/.py//g" | sed -e "s/scripts\/ja-edited\///g"| xargs -n1 -I{} jupytext --output ja-edited/{}.ipynb scripts/ja-edited/{}.py