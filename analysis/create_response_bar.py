from evaluation_metrics import EvaluationMetrics
from joblib import Parallel, delayed
import pandas as pd
import altair as alt
import numpy as np
import ast
from sentence_transformers import util
import torch
from torch import tensor
import os

AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"

class ResponseBar:
    def __init__(
        self, 
        test_type, 
        prompt_type,
        model_config,
        prompt_type_2='instructions_question',
        top_p_dir='p04', 
        temperature_dir='t1', 
        
    ):
        self.questions = pd.read_csv(f'{AWS_PREFIX}/{test_type}/questions.csv')
        self.top_p_dir = top_p_dir
        self.temperature_dir = temperature_dir
        self.test_type = test_type
        self.prompt_type = prompt_type
        self.model_config = model_config

        def clean_prompt(r):
            r = str(r)
            r = r.replace("<image>", "").replace("\n", "").replace(",", "")
            r = r.lower()
            r = r.replace("user: ", "user:  ")
            return r

        model_response_url = f'{AWS_PREFIX}/{test_type}/responses/{prompt_type}/{top_p_dir}/{temperature_dir}/model_responses.csv'
        self.model_responses = pd.read_csv(model_response_url)
        self.model_responses = self.model_responses.rename({
            "imageFile": "image_file"
        }, axis=1)
        self.model_responses = self.model_responses[self.model_responses["agentType"] == model_config[0]]
        self.model_responses["prompt"] = self.model_responses["prompt"].apply(clean_prompt)
        self.model_responses["agent_response"] = self.model_responses.apply(
            lambda r : str(r["agent_response"]).replace(r["prompt"], ""), 
            axis=1
        )
        self.model_responses["agent_response"] = self.model_responses["agent_response"].replace({ "": np.nan, "nan": np.nan })

        if prompt_type_2:
            model_response_url_2 = f'{AWS_PREFIX}/{test_type}/responses/{prompt_type_2}/{top_p_dir}/{temperature_dir}/model_responses.csv'
            self.model_responses_2 = pd.read_csv(model_response_url_2)
            self.model_responses_2 = self.model_responses_2.rename({
                "imageFile": "image_file"
            }, axis=1)
            self.model_responses_2 = self.model_responses_2[self.model_responses_2["agentType"] == model_config[0]]

            self.model_responses_2["prompt"] = self.model_responses_2["prompt"].apply(clean_prompt)
            self.model_responses_2["agent_response"] = self.model_responses_2.apply(
                lambda r : str(r["agent_response"]).replace(r["prompt"], ""), 
                axis=1
            )
            self.model_responses_2["agent_response"] = self.model_responses_2["agent_response"].replace({ "": np.nan, "nan": np.nan})
        


    def create_non_empty_response_bar(self):
        
        na_count = self.model_responses["agent_response"].isna().sum()
        non_na_count = self.model_responses["agent_response"].notna().sum()

        raw_na_count = self.model_responses_2["agent_response"].isna().sum()
        raw_non_na_count = self.model_responses_2["agent_response"].notna().sum()

        data = pd.DataFrame({
            'prompt_type': ['raw', 'adapted'],
            'count': [raw_non_na_count / (raw_na_count + raw_non_na_count),
                    non_na_count / (na_count + non_na_count)]
        })
        _dir = f"./non_null_df/{self.model_config[0].replace("/", "_")}"
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # data.to_csv(f"{_dir}/{self.test_type}.csv")

        print(f"NA Count {na_count}, Non NA Count {non_na_count}, Raw NA Count {raw_na_count}, Raw Non NA Count {raw_non_na_count}")

        yaxis = None if self.test_type != "ggr" else alt.Axis(labels=True, ticks=False, domain=False, title=None)

        chart = alt.Chart(data).mark_bar(color=self.model_config[1]).encode(
            x=alt.X('prompt_type', title=None, axis=None, scale=alt.Scale(domain=["raw", "adapted"])),
            y=alt.Y('count', title=None, scale=alt.Scale(domain=[0, 1]), axis=yaxis ),
            color=alt.Color(legend=None),
            opacity=alt.Opacity('prompt_type:N',  legend=None,
                                scale=alt.Scale(domain=["raw", "adapted"], range=[0.5, 1]))
        ).properties(width=30, height=100, title=alt.Title(
            self.test_type,
            anchor="middle"
        ))

        return chart


def non_empty_response_bar():
    prompt_type = "indist_instructions_question"

    model_configs = [
        {'model': "Salesforce/blip2-flan-t5-xl",  'top_p': 'p06', 'temp': 't1'}, 
        {'model': "Salesforce/blip2-flan-t5-xxl",  'top_p': 'p1', 'temp': 't10'},
        {'model': "llava-hf/llava-1.5-7b-hf",  'top_p': 'p04', 'temp': 't1'},
        {'model': "llava-hf/llava-1.5-13b-hf",  'top_p': 'p1', 'temp': 't04'},
        {'model': "liuhaotian/llava-v1.6-34b",  'top_p': 'p1', 'temp': 't04'}, 
        {'model': "google/pix2struct-chartqa-base",  'top_p': 'p08', 'temp': 't1'},
        {'model': "google/matcha-chartqa",  'top_p': 'p04', 'temp': 't1'},
        {'model': "GPT-4V",  'top_p': 'p1', 'temp': 't02'}, # need to run -- 6x more
    ]

    all_charts = []
    for model_config in model_configs:
        model_charts = []
        model = model_config['model']
        top_p_dir = model_config['top_p']
        temperature_dir = model_config['temp']
        for test_type in [
            "ggr",
            "vlat",
            "holf",
            'calvi-trick',
            'calvi-standard',
            'holf2',
            'chartqa-val',
            'chartqa-test'
            'chartqa-test-continuous',
        ]:   
            response_bar = ResponseBar(
                test_type=test_type, 
                prompt_type=prompt_type,
                model_config=model,
                top_p_dir=top_p_dir,
                temperature_dir=temperature_dir
            )
            chart = response_bar.create_non_empty_response_bar()
            model_charts.append(chart)
            print(model, test_type)
        
        chart = alt.hconcat(*model_charts, spacing=2)
        all_charts.append(chart)
        file_name = f"./bar_plots/response_bar_{model[0].replace("/", "-")}.pdf"
        print(file_name)
        chart.save(file_name)
    # alt.hconcat(*all_charts).save("./bar_plots/response_bar_all_models.pdf")

if __name__ == '__main__':
    non_empty_response_bar()
        
    
    
    

