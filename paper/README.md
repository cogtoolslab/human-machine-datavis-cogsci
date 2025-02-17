Two errors have been identified and corrected in the original version of the paper: `vlm_datavis_benchmark_cogsci2024.pdf`. The corrected version can be found in the following pdf: `vlm_datavis_benchmark_corrected_cogsci2024.pdf`.

## Errata

## Subsection: Model evaluation procedure, Paragraph 2.
**Orignal text**: "We sampled outputs from each model using nucleus sampling (temperature = $0.1$; top-$p$ = 0.4), a commonly used technique for improving the diversity and fluency of language model outputs"

**Corrected text**: "We sampled outputs from each model using nucleus sampling (temperature = $1$; top-$p$ = 0.4), a commonly used technique for improving the diversity and fluency of language model outputs"

**Reasoning**: Temperature used during model evaluation was 1 instead of 0.1

## Figure 3
**Orignal text**: "Human and model performance on each test. Each dot represents either the mean proportion correct (GGR & VLAT)
or median normalized absolute error (VLAT) for a single test item. Error bars represent bootstrapped 95% confidence intervals"

**Corrected text**: "Human and model performance on each test. Each dot represents either the mean proportion correct (GGR & VLAT)
or median normalized absolute error (HOLF) for a single test item. Error bars represent bootstrapped 95% confidence intervals"

**Reasoning**: Each dot represents the median normalized absolute error in the HOLF plot only.


