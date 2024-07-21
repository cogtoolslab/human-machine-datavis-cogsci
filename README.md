# Evaluating human and machine understanding of data visualizations

This repository contains code to reproduce the results in our CogSci 2024 paper, *Evaluating human and machine understanding of data visualizations*

Directory Structure

```bash
├── admin
├── paper
├── analysis
├── experiments
│   ├── api  
│   ├── model_inference
├── results
│   ├── dataframe
│   ├── figures
├── stimuli
│   ├── ggr
│   ├── vlat
│   ├── holf
```

`paper` contains the pdf for the orginial and corrected version our paper.

`analysis` contains 3 main files -

3. `environment.yml` specifies a conda environment with the appropriate packages to reproduce our code. We recommend creating a new conda environment using the following command:
   ```
   conda env create -f environment.yml
   ```
`experiments` contains 2 subfolders -
`category-selfpaced` contains code and materials for the human sketch production experiment
`recognition` contains code and maetrials for the human sketch recognition experiment

`results` contains a subdirectory called `plots` which is where the jupyter notebooks will save plots that are generated.

`data` will need to contain intermediate outputs, which serve as input for the R markdown and jupyter notebooks. Please place the contents found inside the `recog_exp_data` folder <a href="https://www.dropbox.com/scl/fo/2oqncsagow0k7sbn52pd1/h?dl=0&rlkey=i7ezf9lezft7o0amawb24zlvd" target="_blank">here</a>
 inside the `data` directory running any notebook cells. Refer to the README.md in the `data` directory for more details.

