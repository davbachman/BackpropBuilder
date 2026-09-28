# Standardization and California housing L1 pilot

The intended exercise works with three retained features: median income, latitude, and longitude. Retaining only two noticeably worsens prediction.

## Using the block

Add **Standardize features** between the assembled input vector and the first trainable matrix product. Select it and click **Fit on training rows**. Each feature becomes (x − mean) / population standard deviation, using only training examples. The inspector shows the fitted means, scales, and observation count. Constant columns use scale 1.

Statistics are fixed during training, held-out inference, and prediction, and included in saved projects and Python exports. Resetting model weights preserves them. Refit explicitly after changing features, dataset, or training split; never refit on a new test set. The block accepts scalar inputs or features along the last tensor axis. It rejects fitting from targets or trainable preprocessing. Use standardization after creating polynomial terms.

Numeric CSV imports now allow 50,000 rows (still 10 MB and 129 columns). Numeric examples are cached, and large datasets use a numeric example selector instead of thousands of dropdown options.

## Dataset and protocol

The [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) contains 20,640 census block groups and eight numeric predictors. The response is median house value in units of $100,000, not an individual house price.

The seeded random split uses 14,448 training rows, 3,096 validation rows, and 3,096 final test rows. The classroom CSV contains training and validation only; its explicit “test” rows serve as validation in the app. Final test data remain separate. These are random splits, not a test of geographic extrapolation.

All input scaling was fitted by the actual app block on training rows. The target was left in its original units. Models use an explicit matrix product plus bias and MSE, with no prebuilt regression estimator.

Validation explored L1 strengths 0, .01, .05, .1, .2; the baseline and chosen .05 recipe were repeated with initialization seeds 137, 211, and 307. Three-feature and two-feature refits also used these seeds. The final twelve comparison recipes and file hashes were frozen before inspecting final test performance.

## Recommended settings

- Optimizer: AdamW; learning rate .03; weight decay 0; gradient clipping 0.
- Full training batch: 14,448 examples; maximum 1,500 epochs.
- Early stopping: validation data loss, patience 50, minimum improvement .00001, restore best weights.
- L1 strength: .05, weights only, excluding bias.
- Penalty: lambda times the sum of absolute weights, added to mean squared error.
- Select the three largest absolute standardized coefficients, then rebuild with those inputs and L1 set to zero.

With seed 137, the L1 fit restored epoch 126. Approximate coefficients in CSV order were .723, .138, −.002, .038, .005, −.009, −.631, −.586. All three seeds selected MedInc, Latitude, Longitude. L1 with the current subgradient optimizer encourages small coefficients but does not promise exact zeros.

When importing an instructor project, set **Epochs per run = 1500** and **Examples per update = 14448** manually; these run controls are not saved in project JSON.

## Frozen final-test results

Means over three initialization seeds; RMSE is converted to dollars.

| Model | Test MSE ($100,000 units squared) | Test RMSE |
|---|---:|---:|
| Eight features, no penalty | .54216 | $73,632 |
| Eight features, L1 .05 | .55277 | $74,348 |
| Three selected features, no penalty | .54288 | $73,680 |
| Two selected features, no penalty | .68305 | $82,647 |

The three-feature refit is essentially tied with the full unregularized model. Two features are insufficient here. Coefficients reflect conditional predictive associations; correlated inputs and the joint geographic contribution mean their ranking is not a causal importance ranking.

## App verification and reproduction

The full L1 model trained from scratch using the actual WebGL training engine in about 55 seconds, restoring epoch 126 with validation MSE .5384668, matching the native pilot. Browser checks recompute fitted statistics and compare initial loss and trained-model validation loss. The main app's inspector was also exercised on the full raw dataset, including refitting, importing the reduced model, tracing forward, and evaluating all 3,096 validation examples (MSE .54417). WebGL parity checks passed for the full baseline, L1, two-feature, and three-feature models. Browser evidence is saved in output/housing-pilot/browser/.

Automated coverage includes training-only fitting with held-out outliers, constant columns, scalar/batch transforms, trace/tensor gradient agreement, saved-project and export behavior, invalid inputs, full-size CSV acceptance, and dataset cache invalidation.

Scripts, in order: prepare-housing-pilot.py, prepare-housing-models.mjs, train-housing-pilot.py, reduce-housing-models.mjs, train-housing-pilot.py, finalize-housing-pilot.py, export-housing-models.mjs. Run Python scripts with an environment containing scikit-learn and PyTorch, and JavaScript scripts with Node from the repository root. Scripts currently use the fixed output/housing-pilot path. Finalization locks the selected recipes; reproduce in a separate checkout/output tree rather than overwriting the frozen pilot. serve-housing-pilot.mjs serves the browser verification page.

Generated CSVs, initial/trained instructor projects, raw runs, frozen hashes, and browser evidence stay git-ignored under output/housing-pilot/. Start with its START-HERE.md. These are instructor references, not student starter solutions or grading thresholds.
