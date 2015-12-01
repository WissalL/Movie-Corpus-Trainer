# Movie-Corpus-Trainer

## Running the trainer

* Get [Skip-Thoughts](https://github.com/ryankiros/skip-thoughts)

* Get [Cornell Movie Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html)

* Put the corpus in `data/cornell-movie-dialogs-corpus`

* We have to make some changes to Skip-Thoughts so we can include it as a module:

	* change the top-level folder name to `skipthoughts`

	* Add an `__init__.py` file to `skipthoughts/` and `skipthoughts/decoding/`

	* Change line 7 of `skipthoughts/decoding/homogeneous_data.py` to `from skipthoughts import skipthoughts`

* Write a file to your home directory called `.theanorc` and put in it:

```
[global]
floatX = float32
```