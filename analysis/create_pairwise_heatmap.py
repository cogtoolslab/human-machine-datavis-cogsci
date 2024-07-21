from evaluation_metrics import EvaluationMetrics
from joblib import Parallel, delayed
import pandas as pd
import altair as alt
import numpy as np
from sentence_transformers import util
import torch
from torch import tensor

AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"

class PairwiseHeatmap:
    def __init__(
        self, 
        test_type, 
        prompt_type, 
        units_of_measure = 'a_in_b',
        top_p_dir='p04', 
        temperature_dir='t1',
        model_config=[],
        source="model_responses"
    ):
        self.source = source
        self.evaluation_metrics = EvaluationMetrics()
        self.questions = pd.read_csv(f'{AWS_PREFIX}/{test_type}/questions.csv')
        self.top_p_dir = top_p_dir
        self.temperature_dir = temperature_dir
        self.test_type = test_type
        self.prompt_type = prompt_type
        self.model_config = model_config
        self.units_of_measure = units_of_measure

    def calculate_betweem_prompt_matrix(self, agents, prompt_type_2="instructions_question"):
        def clean_prompt(r):
            r = str(r)
            r = r.replace("<image>", "").replace("\n", "").replace(",", "")
            r = r.lower()
            r = r.replace("user: ", "user:  ")
            return r

        top_p_dir = self.top_p_dir
        temperature_dir = self.temperature_dir
        prompt_type = self.prompt_type
        test_type = self.test_type

        model_response_url = f'{AWS_PREFIX}/{self.test_type}/responses/{self.prompt_type}/{self.top_p_dir}/{self.temperature_dir}/{self.source}.csv'
        print(model_response_url, "model_response_url")
        self.model_responses = pd.read_csv(model_response_url)
        self.model_responses = self.model_responses.rename({
            "imageFile": "image_file"
        }, axis=1)
        self.model_responses = self.model_responses[self.model_responses["agentType"].isin(agents)]

        self.model_responses["prompt"] = self.model_responses["prompt"].apply(clean_prompt)
        self.model_responses["agent_response"] = self.model_responses.apply(
            lambda r : str(r["agent_response"]).replace(r["prompt"], ""), 
            axis=1
        )
        self.model_responses["agent_response"] = self.model_responses["agent_response"].replace({ "": np.nan, "nan": np.nan })

        # fix this the the processing state but selecting only test
        self.model_responses = self.model_responses[self.model_responses["testType"] == test_type]

        model_response_url_2 = f'{AWS_PREFIX}/{test_type}/responses/{prompt_type_2}/{top_p_dir}/{temperature_dir}/{self.source}.csv'
        print(model_response_url_2, "model_response_url_2")
        self.model_responses_2 = pd.read_csv(model_response_url_2)
        self.model_responses_2 = self.model_responses_2.rename({
            "imageFile": "image_file"
        }, axis=1)
        self.model_responses_2 = self.model_responses_2[self.model_responses_2["agentType"].isin(agents)]

        self.model_responses_2["prompt"] = self.model_responses_2["prompt"].apply(clean_prompt)
        self.model_responses_2["agent_response"] = self.model_responses_2.apply(
            lambda r : str(r["agent_response"]).replace(r["prompt"], ""), 
            axis=1
        )
        self.model_responses_2["agent_response"] = self.model_responses_2["agent_response"].replace({ "": np.nan, "nan": np.nan})
        self.model_responses_2 = self.model_responses_2[self.model_responses_2["testType"] == test_type]

        if self.source == "model_responses":
            self.model_responses["iteration_count"] = self.model_responses.groupby(
                ["agentType", "question", "image_file"]
            ).cumcount() + 1
            self.model_responses_2["iteration_count"] = self.model_responses.groupby(
                ["agentType", "question", "image_file"]
            ).cumcount() + 1
        
        if self.source == "processed_extracted_responses":
            print("Model responses loaded, adding humans responses...")
            raw_model_response_url = f'{AWS_PREFIX}/{test_type}/responses/{prompt_type}/{top_p_dir}/{temperature_dir}/model_responses.csv'
            self.human_responses = pd.read_csv(raw_model_response_url)
            self.human_responses = self.human_responses[(self.human_responses["agentType"] == "Human/Math-2-1") | 
                                                        (self.human_responses["agentType"] == "Human/Math-3")] #| 
                                                        # (self.human_responses["agentType"] == "Human")]

            self.model_responses = pd.concat([self.model_responses, self.human_responses])
            self.model_responses["iteration_count"] = self.model_responses.groupby(
                ["agentType", "question", "image_file"]
            ).cumcount() + 1
            self.model_responses_2 = self.model_responses.copy()

            # self.model_responses_2 = pd.concat([self.model_responses_2, self.human_responses])

            # self.model_responses_2 = self.model_responses_2

        if test_type == "holf" and self.source != "model_responses":
            subset = [ "question", "image_file", "agentType", "agent_response", "iteration_count"]
            common_responses = self.model_responses.dropna(subset="agent_response")[subset].merge(
                self.model_responses_2.dropna(subset="agent_response")[subset],
                suffixes=('_A', '_B'),
                on=["image_file", "question"]).dropna(
                    subset=["agent_response_A", "agent_response_B"]
                )

            common_responses = common_responses[
                (common_responses["iteration_count_A"] != common_responses["iteration_count_B"])
            ]
            common_responses = common_responses.drop_duplicates(
                subset=["agentType_A", "agentType_B", "question", "image_file", "iteration_count_A", "iteration_count_B"]
            )

            merged_response = self.questions[["question", "image_file", "min_label", "max_label"]]
            common_responses = common_responses.merge(merged_response, on=["question", "image_file"])

            vectorized_get_absolute_error = np.vectorize(self.evaluation_metrics.get_absolute_error)
            common_responses["error"] = vectorized_get_absolute_error(common_responses["agent_response_A"], common_responses["agent_response_B"])
            common_responses["error"] = common_responses["error"]
            common_responses = common_responses.dropna(subset=["error"])

            vectorized_minmax_normalized_error = np.vectorize(self.evaluation_metrics.minmax_normalized_error)
            normalized_errors = vectorized_minmax_normalized_error(
                common_responses["error"], common_responses["min_label"], common_responses["max_label"]
            )

            # agents = common_responses[["agentType_B", "agentType_A"]].copy()
            # agents[self.units_of_measure] = normalized_errors
            common_responses[self.units_of_measure] = normalized_errors
            if self.source == "model_responses":
                common_responses.to_csv(f"./heatmap/between_prompt_pairwise/{self.test_type}_all_pairwise.csv")
            elif self.source == "processed_extracted_responses":
                common_responses.to_csv(f"./heatmap/{self.test_type}_all_pairwise.csv")

            df = common_responses.groupby(["agentType_B", "agentType_A"])[self.units_of_measure].median().reset_index()

            # common_responses["error"] = common_responses.apply(
            #     lambda r: self.evaluation_metric.get_absolute_error(r['agent_response_A'], r['agent_response_B']), axis=1
            # ).dropna(subset=["error"])

            # common_responses[self.units_of_measure] = common_responses.apply(
            #     lambda r: self.evaluation_metric.minmax_normalized_error(r['error'], r['min_label'], r['max_label']), axis=1
            # )

            # print("calculating jaccard similarity")
            # jaccard_similarity = np.vectorize(self.evaluation_metrics.jaccard_similarity)
            # print("calculating jaccard similarity2")
            # jaccard_similarities = jaccard_similarity(prompt_1_responses, prompt_2_responses)
            # print("calculating jaccard similarity3")
            # agents = unique_responses_first[["agentType_B", "agentType_A"]].copy()
            # agents[self.units_of_measure] = jaccard_similarities
            # print("calculating jaccard similarity4")
        else:
            print("merging..")
            subset = ["question", "image_file", "agentType", "agent_response", "iteration_count"]

            common_responses = self.model_responses[subset].dropna(subset="agent_response").merge(
                self.model_responses_2[subset].dropna(subset="agent_response"),
                suffixes=('_A', '_B'),
                on=["image_file", "question"]).dropna(
                    subset=["agent_response_A", "agent_response_B"]
                )
            
            # no need to compare accross same iteration count
            common_responses = common_responses[
                (common_responses["iteration_count_A"] != common_responses["iteration_count_B"])
            ]
            common_responses = common_responses.drop_duplicates(
                subset=["agentType_A", "agentType_B", "question", "image_file", "iteration_count_A", "iteration_count_B"]
            )

            # unique_responses = common_responses.groupby(["question", "image_file", "agentType_A", "agentType_B"])
            # unique_responses_first = unique_responses.first().reset_index()
            # unique_responses_first = common_responses

            print("stacking")
            prompt_1_responses = np.vstack(common_responses['agent_response_A'])
            prompt_2_responses = np.vstack(common_responses['agent_response_B'])

            print("calculating jaccard similarity")
            jaccard_similarity = np.vectorize(self.evaluation_metrics.jaccard_similarity)
            print("calculating jaccard similarity2")
            jaccard_similarities = jaccard_similarity(prompt_1_responses, prompt_2_responses)
            print("calculating jaccard similarity3")
            # agents = common_responses[["agentType_B", "agentType_A"]].copy()
            
            # agents[self.units_of_measure] = jaccard_similarities
            common_responses[self.units_of_measure] = jaccard_similarities

            if self.source == "model_responses":
                common_responses.to_csv(f"./heatmap/between_prompt_pairwise/{self.test_type}_all_pairwise.csv")
            elif self.source == "processed_extracted_responses":
                common_responses.to_csv(f"./heatmap/{self.test_type}_all_pairwise.csv")

            print("calculating jaccard similarity4")
            

            df = common_responses.groupby(["agentType_B", "agentType_A"])[self.units_of_measure].mean().reset_index()
        df = df.rename(columns={"agentType_B": "agent_type_2", "agentType_A": "agent_type_1"})
        print("SAVING AGENTS")
        # df.to_csv(f"./heatmap/between_prompt_pairwise/{self.test_type}_pairwise.csv")
        print("calculating jaccard similarity5")

        return df
    
    def create_pairwise_agent_heatmap(self, df):
        # df["cosine_similarity"] = df["cosine_similarity"].apply(lambda x: eval(x).numpy()[0][0])
        # df = df.groupby(["agent_type_1", "agent_type_2"])["cosine_similarity"].mean().reset_index()

        base = alt.Chart(df).mark_rect().encode(
            x=alt.X("agent_type_1", scale=alt.Scale(domain=[model[0] for model in self.model_config])),  
            y=alt.Y("agent_type_2", scale=alt.Scale(domain=[model[0] for model in self.model_config])),
        )

        if self.test_type == "holf":
            color_domain=[0,1.2]
            color_reverse = True
            color_condition = alt.condition(
                alt.datum[self.units_of_measure] > 0.65,
                alt.value('black'),
                alt.value('white')
            )
        else:
            color_domain=[0,1]
            color_reverse = False
            color_condition = alt.condition(
                alt.datum[self.units_of_measure] < 0.65,
                alt.value('black'),
                alt.value('white')
            )

        
        height=150
        width=200 
        title=f"{self.test_type} {self.units_of_measure} between Agents"

        heatmap = base.mark_rect().encode(
            color=alt.Color(f'{self.units_of_measure}:Q', 
                            legend=None, 
                            scale=alt.Scale(scheme="blues", 
                                            domain=color_domain,
                                            reverse=color_reverse)),
        )
        
        text = base.mark_text(baseline='middle').encode(
            alt.Text(f'{self.units_of_measure}:Q', format=".2f"),
            color=color_condition
        )
        
        chart = heatmap.properties(
            width=width,
            height=height,
            title=title
        ) + text

        return chart

if __name__ == '__main__':
    prompt_type = "indist_instructions_question"
    model_sets = [
        ('llava-hf/llava-1.5-7b-hf', '#f9b574', '#f9b574'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#5ba3cf'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#4c78a8'),
        ('GPT-4V', '#b85536', '#b85536'),
        ('Human/Math-2-1', "#639460", "#C8EBC6"),
        ('Human/Math-3', "#2e693b", "#73D287"),
    ]

    for test_type in [
        "ggr", 
        "vlat", 
        "holf"
    ]:  
        # heatmap = PairwiseHeatmap(
        #     test_type=test_type, 
        #     prompt_type=prompt_type,
        #     model_config=model_sets,
        #     units_of_measure="jaccard_similarity",
        #     source="model_responses"
        # )
        # heatmap.calculate_betweem_prompt_matrix(
        #     agents=[m[0] for m in model_sets],
        #     prompt_type_2 = "instructions_question"
        # )
        heatmap = PairwiseHeatmap(
            test_type=test_type, 
            prompt_type=prompt_type,
            model_config=model_sets,
            units_of_measure="jaccard_similarity",
            source="processed_extracted_responses"
        )
        
        print("Creating matrix...")
        matrix = heatmap.calculate_betweem_prompt_matrix(
            agents=[m[0] for m in model_sets], 
            prompt_type_2="indist_instructions_question")
        print("Matrix created")
        
        _df_dir = "../results/dataframe/pairwise_heatmap_df"
        file = f"{_df_dir}/pairwise_heatmap_df_{test_type}.csv"
        matrix.to_csv(file)

        # print("Creating heatmap...")
        chart = heatmap.create_pairwise_agent_heatmap(matrix)
        _dir = "../results/figures/pairwise_heatmap"
        filename = f"{_dir}/pairwise_heatmap_{test_type}.pdf"
        print("Saving...", filename)
        chart.save(filename)
        # print("Saved Heatmap")