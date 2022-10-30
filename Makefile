main: nb2script

nb2script:
	echo "Converting the notebook to a python script"
	jupyter nbconvert --to script notebook.ipynb
