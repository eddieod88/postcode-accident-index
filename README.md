# Postcode Risk Nearest Neighbour Estimator

Postcode districts often have risk ratings which are used as a key factor in motor insurance pricing models.  But what happens when new postcodes are created (through new housing estates being built) but your **external** postcode ratings table has not been updated?  If a quote comes through with one of these new postcodes, it would be good if we could estimate the geographical risk based on our existing knowledge automatically.  It is logical to think that nearby postcodes should have a similar geographical risk and so a good starting point could be to estimate the risk of the postcode by using a nearest neighbour approach.

## Setup

This projects uses Python and Poetry as a package/environment manager:

```bash
poetry
python >= 3.10.1
```

run `poetry install` in the root directory of this project to install the environment from the `poetry.lock` file, then run `poetry shell` to activate the environment.

## Data

https://www.kaggle.com/datasets/manishtripathi86/predict-accident-risk-score-for-unique-postcode

## Methods

### 1 - Generate bucketed rankings for the postcode (districts)

Bit of feature engineering and then probably a GBT or random forest due to the size of the data to predict the accident index.

- will need to predict the total casualties at a postcode in a given year and also the total accidents in a year, then divide the two (or do em together.)

divide the rankings into 20 or 50 percentiles such that we have ordinal buckets of equal size -> this will be the postcode risk ranking

### 2 - Immitate the use case where a postcode is new and has an unknown ranking

A postcode comes in and is checked against our existing mapping (postcode -> ranking).  If the postcode is new, it will use a postcode API to fetch the coords for the postcode and apply a NN regression, weighted by distance to the nearest existing postcodes and see how close it is to the true ranking.


## Outcomes

...

## Notes

Modelling:
- Prediction will take a mode category selector if we do not know the value of a categorical variable present

## TODO:

[X] Evaluation script:
    - Takes in:
        - a folder with:
            - config (with model hyperparameters)
            - predictions (y_true and y_pred) which have been saved due to the use of cross validations
    - Process:
        - generates outputs based on the config file (which metrics, which visualisations...)
    - Outputs (in same folder as the config and model):
        - metrics and visualisations. 
[ ] dvc the training to evaluation together
[ ] Got some prelim results in a neatish format.  Now need to look into the following:
    - would it be better if we make this a classification task as we can't predict the extreme values
    - try removing variables such as count?
    - try more simple regression modelling techniques

[ ] Try a better joining strategy for the postcode to pop & area datasets (ie a postcode fall back mechanism)... this may make some of the nearest road stats completely wrong though...