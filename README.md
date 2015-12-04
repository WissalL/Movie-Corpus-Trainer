# Movie-Corpus-Trainer

## Running the trainer

Put my fork of skip-thoughts inside your clone of this repo:
```
git clone https://github.com/TManzini/Movie-Corpus-Trainer.git
cd Movie-Corpus-Trainer
git clone https://github.com/jongoodnow/skip-thoughts
```

Change the name of the `skip-thoughts` directory to `skipthoughts`.

Your directory structure should look like this:
```
Movie-Corpus-Trainer/
	data/
	skipthoughts/
		data/
		decoding/
		training/
		...
	demo.py
```

Put the data from the skip-thoughts readme into `Movie-Corpus-Trainer/skipthoughts/data`. Put the [Cornell Movie Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html) in `Movie-Corpus-Trainer/data/cornell-movie-dialogs-corpus`. Your other dialog will go in this data folder.

Write a file to your home directory called `.theanorc` and put in it:
```
[global]
floatX = float32
```