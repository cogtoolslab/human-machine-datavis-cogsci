All analysis scripts can be run without download any data locally. If you would like to download the data for model responses, please find publicly-hosted files on the Cognitive Tools Lab AWS server below:

All raw model responses can be found at:
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/model_responses.csv
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/model_responses.csv
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/model_responses.csv

All valid model responses can be found at:
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv
- https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv

Please see `/experiments/model_inference/api/azure_gpt4_template.py` for GPT-4 prompt and API to extract valid model responses. Please see `/analysis/process_extracted_responses.py` for processing procedure for valid responses.