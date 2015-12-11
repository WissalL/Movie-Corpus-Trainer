# Movie-Corpus-Trainer

This is a chat bot that lets you talk to a movie or television character, given scripts of that character's conversations. For this demo, we use the character Fry from the television series Futurama.

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

Put the data from the skip-thoughts readme into `Movie-Corpus-Trainer/skipthoughts/data`. Put your script data in the `Movie-Corpus-Trainer/data` directory. For this demo, we use `fry_sources.txt` and `fry_targets.txt`, where sources are lines said to Fry, and targets are his responses to those lines.

Write a file to your home directory called `.theanorc` and put in it:
```
[global]
floatX = float32
```

Run:
```
python build.py
```
It will take multiple hours to finish.

## Running the chat bot

```
python chat.py
```
When you send messages, your punctuation should be separated from the words. For example: `Who are you ?`.