import pandas as pd
import altair as alt
from utils import read_model_configs, create_all_agent_response_dataframe, get_all_test_types
import numpy as np
import math


def prepare_dataframe(test_type):
    model_configs = read_model_configs()[:-1]
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False
    )

    return model_responses[['question', 'image_file', 'agent_response']].dropna()


if __name__ == '__main__':
    test_types = get_all_test_types()
    all_dfs = []
    for test in test_types:
        all_dfs.append(prepare_dataframe(test))

    all_dfs = pd.concat(all_dfs).value_counts().reset_index()
    file = "./valid_value_counts.csv"
    print(file)
    all_dfs.to_csv(file)