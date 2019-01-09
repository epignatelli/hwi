[![Slack Status](https://slack.bhom.com/badge.svg)](https://bhom.xyz)

### Get started

1. Download and install [anaconda python](https://www.anaconda.com/download/) and make sure to add it to the environmental path

2. Open a command line tool and create a new environment from the conda-envs/ml.yml file:
`conda env create --name ml -f=ml.yml`


### Get the data

1. Download the data from [here](https://www.kaggle.com/c/6818/download-all) and place it in a `data` folder in the main directory.
The folder structure will look like this:
```
- src
- data
    |
    - train
        - 0000e88ab.jpg
        -...
    - test
        - 000dcf7d8.jpg
        - ...
```
2. Download the nasnet weights by openin the `download-weights.bat` file in the src folder


### Running the code

1. open a command line tool pointing to the `src` folder and run the command:
`python main.py`

Or use PyCharm and click on the run button
