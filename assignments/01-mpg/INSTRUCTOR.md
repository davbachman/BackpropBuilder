# Assignment 1 — MPG regression (mpg-01-v3)

This is a ready-to-review first assignment after the in-class one- and two-dimensional regression demonstrations. Students build all models themselves, including Standardize Features blocks fitted on training rows. No prebuilt student models or notebooks are supplied.

## Files to post

Generated deliverables are under ../../output/assignment-01-mpg/ and remain git-ignored.

1. Paste the contents of **student-files/canvas.html** into Canvas's HTML editor for the assignment description. It is an HTML fragment, not a scripted app. Set your own dates and Gradescope link.
2. Attach **student-files/mpg.csv**, and insert a link to it in the Files paragraph.
3. Alternatively, attach **canvas-student-files.zip** and tell students to extract it. That archive contains only the instruction HTML and CSV.
4. Upload **gradescope-mpg-01-v3.zip** only to the Gradescope autograder configuration. Do not distribute the grader or instructor models as student files.

## Gradescope setup

Use a Programming Assignment with a 100-point total. Accept the three model JSON files named in the instructions. Use an Ubuntu 22.04 or 24.04 base image with at least 1 GB memory and a 60-second or longer grading timeout. setup.sh installs Python, creates /opt/mpg-grader, and pins NumPy 1.26.4. No API keys, GPU, paid inference, or student code execution are required.

The ZIP has setup.sh and executable run_autograder at its root, plus runner.py, grader.py, and the canonical dataset. It follows the [Gradescope autograder specification](https://gradescope-autograders.readthedocs.io/en/latest/specs/). The runner writes /autograder/results/results.json and grades each model in a separate bounded process. Missing/broken models do not erase other parts' credit.

Before releasing, use Gradescope's test-submission facility with only the three instructor model files under output/assignment-01-mpg/standardized-models/. A complete reference submission should receive 100/100. Removing two_features.json should yield 70/100: that model is worth 30 points. The autograder has been tested locally with the pinned dependency, but the hosted Gradescope container build has not been executed here.

Feedback and scores are immediately visible; this supports iterative debugging and resubmission. The held-out set is openly treated as validation, not a hidden independent final test. A version change or rubric change requires rebuilding and uploading a new ZIP; a GitHub push does not update Gradescope.

## Learning objectives and rubric

- Construct and scale feature inputs, wire independently trainable coefficients, and train with full-batch gradient descent.
- Use the same split to compare one-input, two-input, and quadratic-feature regression.
- Distinguish a curved function of inputs from a model linear in its coefficients.
- Understand validation and compare one's own saved models.

The one-feature and two-feature models are worth 30 points each: valid file/data 5, form/scaling 15, training threshold 5, validation threshold 5. The quadratic model is worth 40 points: valid file/data 5, form/scaling 15, training threshold 10, validation threshold 10. Only the three model JSON files are submitted; there are no comprehension questions or answer files. No human grading is required.

The grader accepts scalar Arithmetic graphs and equivalent small matrix-product graphs. It symbolically evaluates a restricted polynomial computation, requires the intended polynomial family and independent trainable coefficients, detects target dependencies, and recomputes errors using the canonical CSV. It never trusts cached tensor outputs, saved losses, reported epochs, or completion flags. It checks that each used raw feature has a fixed shrinking affine transformation upstream of parameters (0 < absolute slope < 1). State this interpretation when discussing alternative scalings. No claim is made that exported final parameters prove the student personally performed training.

## Pilot evidence

The CSV uses all 398 Auto MPG rows because only weight and year are required. Raw weight is in pounds; model year is expanded to four digits. The split is fixed by Python Random(20260929): 318 training and 80 validation rows. The input transformations were built inside each app graph:
u = (weight_lb − training mean weight) / training standard deviation of weight; v = (model_year − training mean year) / training standard deviation of year. Each block is fitted on 318 training observations.
The target stays in mpg.

Full-batch SGD, learning rate .05, 1,000 epochs, no regularization, was run with seed 137 for the standardized models (the earlier Arithmetic pilot used seeds 137, 211, 307) using the app's actual trace training engine. An independent NumPy gradient-descent calculation and least-squares solution confirmed convergence.

| Model | App training MSE | App validation MSE | Required train / validation MSE |
|---|---:|---:|---:|
| One feature | 20.2484 | 13.0160 | ≤22 / ≤16 |
| Two features | 12.7574 | 7.5553 | ≤14 / ≤10 |
| Quadratic | 9.9606 | 5.5748 | ≤11 / ≤8 |

The earlier Arithmetic-scaled full 1,000-epoch quadratic run was also executed through the main browser app with the published settings: training MSE 9.961 and validation MSE 5.575. Its actual File → Save download received 40/40 from the current grader.

The three earlier Arithmetic initializations give essentially the same final result; the new standardized pilot agrees to four decimal places. These are feasibility results for this particular fixed split, not a claim that added complexity always improves generalization. The validation split is easier here than the training split; a smaller validation MSE is possible and is not evidence that training/validation were reversed.

Current instructor projects are in standardized-models/ (trained files and initial-*.json). The models/ folder retains the earlier Arithmetic-scaled models for compatibility testing. They are not student starter files. Run controls such as epochs and examples per update must be set after import; the project format does not store those controls.

## Reproduction and verification

From the repository root, with Python 3.12 plus NumPy and npm dependencies installed:

- python scripts/assignments/prepare-mpg.py
- node scripts/assignments/pilot-mpg.mjs
- node scripts/assignments/pilot-mpg-standardized.mjs
- python -m unittest discover -s assignments/01-mpg -p test_grader.py -v
- python scripts/assignments/package-mpg.py

Preparation downloads the UCI source archive, records its SHA-256, and writes the deterministic CSV and independent pilot. Packaging records artifact hashes. Do not regenerate an issued assignment against changed source data or change the split without a new version.

Grader tests cover app/independent-engine parity, alternate scaling and matrix forms, untrained partial credit, ignored cached scores/overrides, target leakage, wrong features, modified data, missing scaling, invalid JSON/values/cycles/expressions/oversized tensors, and full/partial runs in the Gradescope directory layout.

Version mpg-01-v3 uses Standardize Features and adds grading support for its saved statistics. The dataset, thresholds, and 30/30/40 rubric are unchanged. Earlier Arithmetic-scaled models remain accepted. Earlier materials are preserved under output/assignment-01-mpg/archive/. Replace any v1 or v2 Gradescope upload with gradescope-mpg-01-v3.zip. Standardization statistics are checked for valid dimensions and finite positive scales; the grader evaluates the saved transformation and does not certify that a student clicked Fit on training rows.

No application changes are needed for this assignment. No Canvas or Gradescope content has been published, and no files have been committed or pushed by this task.
