# ChessHacks Starter Bot

A set of tools for testing and deploying chessbots, developed by ChessHacks. Our [chessbot](https://github.com/sunray4/chessbot_model/tree/main) is integrated in [./src/main.py](https://github.com/sunray4/chessbot_deploy/blob/main/src/main.py). Setting up the repostiory according to [instructions](#setup) creates a chessboard where you can play against our chessbot with tools to evalute its performance!!

## Directory Structure

`/devtools` is a Next.js app that provides a UI for testing your bot. It includes an analysis board that you can use to test your bot and play against your own bot. You do not need to edit, or look at, any of this code (unless you want to).

`/src` is the source code for your bot. You will need to edit this code to implement your own bot.

`serve.py` is the backend that interacts with the Next.js and your bot (`/src/main.py`). It also handles hot reloading of your bot when you make changes to it. This file, after receiving moves from the frontend, will communicate the current board status to your bot as a PGN string, and will send your bot's move back to the frontend. You do not need to edit any of this code (unless you want to).

While developing, you do not need to run either of the Python files yourself. The Next.js app includes the `serve.py` file as a subprocess, and will run it for you when you run `npm run dev`.

The backend (as a subprocess) will deploy on port `5058` by default.

## Setup

Start by creating a Python virtual environment and installing the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or however you want to set up your Python.

Then, install the dependencies for the Next.js app:

```bash
cd devtools
npm install
```

Afterwards, make a copy of `.env.template` and name it `.env.local` (NOT `.env`). Then fill out the values with the path to your Python environment, and the ports you want to use.

> Copy the `.env.local` file to the `devtools` directory as well.

Lastly, simply run the Nextjs app inside of the devtools folder.

```bash
cd devtools
npm run dev
```

## Troubleshooting

First, make sure that you aren't running any `python` commands! These devtools are designed to help you play against your bot and see how its predictions are working. You can see [Setup](#setup) and [Running the app](#running-the-app) above for information on how to run the app. You should be running the Next.js app, not the Python files directly!

If you get an error like this:

```python
Traceback (most recent call last):
  File "/Users/obama/dev/chesshacks//src/main.py", line 1, in <module>
    from .utils import chess_manager, GameContext
ImportError: attempted relative import with no known parent package
```

you might think that you should remove the period before `utils` and that will fix the issue. But in reality, this will just cause more problems in the future! You aren't supposed to run `main.py ` on your own—it's designed for `serve.py` to run it for you within the subprocess. Removing the period would cause it to break during that step.

### Logs

Once you run the app, you should see logs from both the Next.js app and the Python subprocess, which includes both `serve.py` and `main.py`. `stdout`s and `stderr`s from both Python files will show in your Next.js terminal. They are designed to be fairly verbose by default.


