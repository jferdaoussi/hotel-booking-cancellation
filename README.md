# Hotel Booking Cancellation

Estimate the probability that a hotel booking will be cancelled. This project compares a historical cancellation-rate baseline, logistic regression, random forest and histogram gradient boosting in scikit-learn.

## Result

On 26,565 bookings made from January 2017 onward, the random forest selected in validation reduced log loss from **0.6387** to **0.5350** against the historical-rate baseline.

| Model | Validation log loss ↓ | Holdout log loss ↓ | Brier score ↓ | ROC-AUC ↑ | Average precision ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical cancellation rate | 0.6826 | 0.6387 | 0.2231 | 0.5000 | 0.3163 |
| Logistic regression | 0.5674 | 0.5669 | 0.1925 | 0.6970 | 0.5274 |
| **Random forest · selected in validation** | **0.5385** | 0.5350 | 0.1800 | 0.7426 | 0.5792 |
| Histogram gradient boosting | 0.5522 | 0.5209 | 0.1756 | 0.7542 | 0.5979 |

Gradient boosting scored best on the holdout. The saved model is the random forest, selected before scoring the holdout.

## Run

Requires Python 3.12.

```sh
python -m pip install -r requirements.txt
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv', 'hotels.csv')"
python booking_model.py train hotels.csv
python booking_model.py predict artifacts/model.joblib examples/bookings.csv
```

Training writes the fitted pipeline and scores to `artifacts/`. The example CSV contains two invented bookings, and inference prints one cancellation probability per row. The reported evaluation used the coursework copy of the dataset, with 119,390 rows.

## Evaluation design

- Approximate booking date by subtracting lead time from arrival date.
- Fit on earlier bookings with a known outcome at each training cutoff. Select on bookings made in the second half of 2016 whose outcomes were known by the end of that year.
- Keep bookings made from 1 January 2017 onward for the holdout. Fit imputation, scaling and encoding on the training records only.
- Use an explicit input list. Final reservation status, resolution date, assigned room and post-booking changes stay out of the inputs.
- The data comes from two Portuguese hotels. Booking dates are reconstructed, and the dataset covers a limited arrival window. Scores describe this cohort.

## Origin

This is my independent follow-up to a master's team project. In the team project I [implemented the five-model comparison and hyperparameter search](https://github.com/AdrianOropeza/pontia-ml/commit/0b7ba56). This version implements its own pipeline, temporal evaluation and inference command. [Original team repository](https://github.com/AdrianOropeza/pontia-ml)

Data source: Nuno Antonio, Ana de Almeida and Luís Nunes, [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126), 2019. [TidyTuesday dataset mirror](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md)
