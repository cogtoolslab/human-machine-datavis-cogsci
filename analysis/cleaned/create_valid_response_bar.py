import pandas as pd
import numpy as np
import altair as alt
from aws_upload import upload_file_to_s3
import os
from utils import read_model_configs, extract_numerical_responses

AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"

def process_extracted_responses(
        model_configs, 
        test_types, 
        prompt_type='indist_instructions_question'
    ):

    model_responses = []
    for model_config in model_configs:
        model = model_config['model']
        temperature_dir = model_config['temp']
        top_p_dir = model_config['top_p']
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{model.replace("/", "-")}'
        for test_type in test_types:
            file_name = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/{_dir}/extracted_responses.csv'
            response_file = pd.read_csv(file_name)

            if test_type == 'chartqa-test-continuous-human':
                response_file['testType'] = 'chartqa-test-continuous-human'

            model_responses.append(response_file)

    processed_dfs = pd.concat(model_responses)

    questions = []
    for test_type in test_types:
        file_name = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv"
        questions.append(pd.read_csv(file_name))

    questions = pd.concat(questions)

    def find_non_floats(r):
        if r['res_len'] <= 2:
            res = str(r['agent_response'])
            res = res.replace(" ", "")
            res = res.replace(",", "")
            extracted_response = extract_numerical_responses(res)
            if extracted_response:
                return extracted_response[0]
            else:
                return np.nan
        else:
            return np.nan
    
    def find_non_mc(r):
        item = questions[
            (questions["question"] == r["question"]) &
            (questions["image_file"] == r["image_file"])                
        ].iloc[0]
        response = str(r["agent_response"]).lower()
        mc1, mc2, mc3, mc4 = item["mc1"], item["mc2"], item["mc3"], item["mc4"]

        for mc in [mc4,mc3,mc2,mc1]:
            if mc != np.nan and response.find(str(mc).lower().strip()) != -1:
                return mc
    
        return np.nan

        
    def extract_valid_response(r):
        test_type = r['testType']
        if test_type in ['holf', 'holf2', 'chartqa-test-continuous', 'chartqa-test-continuous-human']:
            return find_non_floats(r)
        elif test_type in ["vlat", 'calvi-trick', 'calvi-standard']:
            return find_non_mc(r)
        elif test_type in ['ggr']:
            item = questions[
                (questions["question"] == r["question"]) &
                (questions["image_file"] == r["image_file"])                
            ].iloc[0]

            if item['text_input_type'] == 'open_ended_question':
                return find_non_floats(r)
            elif item['text_input_type'] == 'multiple_choice_question':
                return find_non_mc(r)
            else:
                return np.nan
    
    processed_dfs = processed_dfs.merge(questions[['image_file','question']])
    processed_dfs['res_len'] = processed_dfs['agent_response'].apply(
        lambda r : len(str(r).split())
    )
    processed_dfs['extracted_response'] = processed_dfs['agent_response'].copy()
    processed_dfs['agent_response'] = processed_dfs.apply(extract_valid_response, axis=1)

    model_responses = []
    for model_config in model_configs:
        model = model_config['model']
        temperature_dir = model_config['temp']
        top_p_dir = model_config['top_p']
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{model.replace("/", "-")}'
        for test_type in test_types:
            mt_df = processed_dfs[
                (processed_dfs['agentType'] == model) &
                (processed_dfs['testType'] == test_type)
            ]
            # if not os.path.exists(_dir):
            #     os.makedirs(_dir)
            # file_name = f'{_dir}/{test_type}_processed_extracted_responses.csv'
            # mt_df.to_csv(file_name,index=False)
            # upload_file_to_s3(file_name, test_type, f'{_dir}/processed_extracted_responses.csv')

    charts = []
    for i, model_config in enumerate(model_configs):
        model = model_config['model']
        main_color = model_config['color']

        agent_responses =  processed_dfs[processed_dfs["agentType"] == model].copy()[["testType", "agent_response"]].reset_index()
        agent_responses['testType'] = agent_responses['testType'].replace({
            "calvi-trick": "calvi",
            "calvi-standard": "calvi"
        })

        agent_responses["is_valid"] = agent_responses["agent_response"].notna().astype(int)
        agent_responses = agent_responses.groupby(["testType"])["is_valid"].aggregate(lambda g : sum(g) / len(g))

        agent_responses = agent_responses.reset_index()
        
        test_order = [
            "ggr",
            "vlat",
            'calvi',
            "holf",
            'holf2',
            'chartqa-test-continuous',
            # 'chartqa-test-continuous-human',
        ]
        # test_order = test_types

        y_axis = None if i != 0 else alt.Axis()

        # _dir = f"./valid_df/{model.replace("/", "_")}"
        # if not os.path.exists(_dir):
        #     os.makedirs(_dir)
        # agent_responses.to_csv(f"{_dir}/{test_type}.csv")

        chart = alt.Chart(agent_responses).mark_bar(color=main_color).encode(
            x=alt.X('testType', title=None, scale=alt.Scale(domain=test_order)),
            y=alt.Y('is_valid', title=None, axis=y_axis, scale=alt.Scale(domain=[0, 1]) ),
            color=alt.Color(legend=None),
        ).properties(width=80, height=100, title=alt.Title(
            "",
            anchor="middle"
        ))
        charts.append(chart)

    file_name = f'./processed_responses_valid.pdf'
    alt.hconcat(*charts, spacing=3).save(file_name)
    print(file_name)



if __name__ == "__main__":
    model_configs = read_model_configs()
    test_types = [
        "ggr",
        "vlat",
        "holf",
    ]

    process_extracted_responses(model_configs=model_configs, test_types=test_types)
    