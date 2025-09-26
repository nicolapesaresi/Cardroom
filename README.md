# Cardroom

This project contains implementations of popular Italian card games environments, with support for reinforcement learning implementations and pygame rendering.

## Installation
The repository is setup us as a poetry project.
To install the repository you can follow these steps:

First, install `poetry` if you haven't already, as indicated by the instructions on the [Poetry installation page](https://python-poetry.org/docs/).
Then, clone the repository to your local machine using the following command:
```
git clone https://github.com/nicolapesaresi/Cardroom.git
cd active-learning
```
Use Poetry to install the project dependencies:
```
poetry install
```
Finally, activate the virtual environment created by Poetry:
```
poetry shell
```

## Games

Implemented games are the following:
- Briscola (two players)
- Briscola (four players) - WIP
- Briscola Chiamata - WIP
- Scopone d'Assi - WIP
- Scopone Scientifico - WIP
- Cotec - WIP

## Agents

Implemented agents are the following:
- Random
- Bot (fixed strategy)
- Information Set Monte Carlo Tree Search
- Human