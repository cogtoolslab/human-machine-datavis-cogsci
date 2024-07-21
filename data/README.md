All analysis scripts can be run without download any data locally. If you would like to download the data for model responses, please find publicly-hosted files on the Cognitive Tools Lab AWS server below:

All human and raw model responses can be found below:

**GGR** 
- Model responses: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/model_responses.csv 
- Human responses: 
    - More math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/pna/tna/Human-Math-3/processed_extracted_responses.csv 
    - Less math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/pna/tna/Human-Math-2-1/processed_extracted_responses.csv

**VLAT** 
- Model responses: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/model_responses.csv
- Human responses: 
    - More math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/pna/tna/Human-Math-3/processed_extracted_responses.csv 
    - Less math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/pna/tna/Human-Math-2-1/processed_extracted_responses.csv

**HOLF** 
-  Model responses: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/model_responses.csv
- Human responses: 
    - More math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/pna/tna/Human-Math-3/processed_extracted_responses.csv 
    - Less math experience: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/pna/tna/Human-Math-2-1/processed_extracted_responses.csv

## Valid model responses
All valid model responses can be found below:
- GGR: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv
- VLAT: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv
- HOLF: https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv

Please see `/experiments/model_inference/api/azure_gpt4_template.py` for the GPT-4 prompt and python API used to extract valid model responses. Please see `/analysis/process_extracted_responses.py` for the processing procedure used on valid responses.