import pandas as pd
import re
import numpy as np
from evaluation_metrics import EvaluationMetrics
from aws_upload import upload_file_to_s3
import os
from joblib import Parallel, delayed

class ModelResponse:

    def __init__(self) -> None:
        pass

    def process_model_responses(self, all_prompts_raw, 
                                agent_type='', 
                                test_type='', 
                                prompt_type='', 
                                temperature=1.0, 
                                top_p=1.0,
                                response_count_per_question=1) -> pd.DataFrame:
        print(f"Processing agent: {agent_type} prompt: {prompt_type}, test: {test_type}, top_p: {top_p}, temperature: {temperature}")
        questions = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv")
        metric = EvaluationMetrics()

        all_prompts = all_prompts_raw.copy()
        all_prompts = self.process_calvi(all_prompts)
        all_prompts = self.process_vlat(all_prompts)
        all_prompts = self.process_ggr(all_prompts)
        # all_prompts = self.process_holf(all_prompts)
        # all_prompts = self.process_chartqa_test(all_prompts)

        given_responses = all_prompts[
            (all_prompts['testType'] == test_type)  &
            (all_prompts['promptType'] == prompt_type) & 
            (all_prompts['agentType'] == agent_type) & 
            (all_prompts['temperature'] == temperature) &
            (all_prompts['topP'] == top_p) 
        ].rename({
            'agentResponse': 'agent_response',
            'isCorrect': 'is_correct',
            'participant': 'participant_id',
            'imageFile': 'image_file'
        }, axis=1)

        responses = []
        for _, test_question_row in questions.iterrows():
            response_row = given_responses[
                (given_responses['question'] == test_question_row['question']) &
                (given_responses['image_file'] == test_question_row['image_file'])          
            ].copy()
            
            if (len(response_row) < 1):
                continue
            
            response_row["agent_response"] = response_row.apply(
                lambda r : metric.remove_prompt(r["agent_response"], r["prompt"]),
                axis=1
            )

            response_row["agent_response"] = response_row.apply(
                lambda r : metric.clean_word(r["agent_response"]),
                axis=1
            )

            response_row = response_row.sort_values(by="timestamp", ascending=False)
            response_row['correct_answer'] = test_question_row['correct_answer']

            if (agent_type.find("Human") != -1):
                if (test_type == "ggr" or test_type == "vlat"):
                    if (agent_type == "Human/Math-2-1"):
                        item_responses = response_row.iloc[0:454]
                    elif (agent_type == "Human/Math-3"):
                        item_responses = response_row.iloc[0:632]
                elif (test_type == "holf"):
                    if (agent_type == "Human/Math-2-1"):
                        item_responses = response_row.iloc[0:284]
                    elif (agent_type == "Human/Math-3"):
                        item_responses = response_row.iloc[0:164]
                elif (test_type == "holf2"):
                    item_responses = response_row.iloc[0:response_count_per_question]
                elif (test_type == "calvi-trick" or test_type == "calvi-standard"):
                    item_responses = response_row.iloc[0:]
                else:
                    item_responses = response_row.iloc[0:response_count_per_question]
            else:
                item_responses = response_row.iloc[0:response_count_per_question]
            responses.append(item_responses)
        
        if (len(responses) == 0):
            print(f"Agent {agent_type} did not respond to any questions for test {test_type}")
            return pd.DataFrame()
        else:
            responses_df = pd.concat(responses)
            if ((len(responses_df) / len(questions)) != response_count_per_question):
                print(f"Agent {agent_type} only responded to {len(responses_df) / len(questions)} questions for test {test_type}")

        
        
        tests_with_continuous_responses = ["chartqa-test-continuous", "holf", "holf2"]
        
        # responses_df["answer_in_response"] = responses_df.apply(
        #     lambda r: metric.a_in_b(r['correct_answer'], r['agent_response']), axis=1
        # )
        # responses_df["raw_response_sbert_embedding"] = responses_df.apply(
        #     lambda r: metric.sbert_embedding(r['agent_response']), axis=1
        # )
        
        
        if (test_type in tests_with_continuous_responses):
            responses_df["relaxed_accuracy"] = responses_df.apply(
                lambda r: metric.relaxed_accuracy(r['agent_response'], r['correct_answer']), axis=1
            )
            responses_df["relaxed_accuracy_e0"] = responses_df.apply(
                lambda r: metric.relaxed_accuracy(r['agent_response'], r['correct_answer'], e=0), axis=1
            )

            responses_df["normalize_by_correct_answer"] = responses_df.apply(
                lambda r: metric.normalize_by_correct_answer(r['agent_response'], r['correct_answer']), axis=1
            )
            responses_df["absolute_error"] = responses_df.apply(
                lambda r: metric.get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
            )
            # response_df["sbert_cos_similarity"] = responses_df.apply(
            #     lambda r: metric.sbert_embedding_cosine_similarity(r['is_correct'], r['agent_response']), axis=1
            # )
          
        else:
            # responses_df["relaxed_accuracy"] = responses_df["answer_in_response"]
            pass
        
        return responses_df
    
    def add_group_metrics(self, responses_df):
        metric = EvaluationMetrics()
        responses_group_stats = responses_df.groupby(['question', 'image_file']).agg(
            group_min=('correct_answer', 'min'),
            group_max=('correct_answer', 'max'),
            group_mean=('correct_answer', 'mean'),
            group_std=('correct_answer', 'std')
        ).reset_index()

        responses_df = responses_df.merge(responses_group_stats, on=['question', 'image_file'], how='left')
        responses_df["one_minus_minmax_normalized_error"] = responses_df.apply(
            lambda r: metric.one_minus_minmax_normalized_error(r['err'], r['group_min'], r['group_max']), axis=1
        )
        responses_df["one_minus_zscore_normalization_error"] = responses_df.apply(
            lambda r: metric.one_minus_zscore_normalization_error(r['err'], r['group_mean'], r['group_std']), axis=1
        )

        return responses_df


    def process_vlat(self, original_df):
        data = original_df.replace({
            'Over the course of years between 2009 and 2014, when was the number of girls named ‘Amelia’ at the maximum?':
            "Over the course of years between 2009 and 2014, when was the number of girls named 'Amelia' at the maximum?",
            "The number of girls named ‘Isla’ was __________ from 2009 to 2012.":
            "The number of girls named 'Isla' was __________ from 2009 to 2012.",
            "Over the course of years between 2009 and 2014, the number of girls named ‘Isla’ was always more than ‘Olivia’.":
            "Over the course of years between 2009 and 2014, the number of girls named 'Isla' was always more than 'Olivia'.",
            "In the UK, the number of girls named ‘Amelia’ in 2014 was more than it was in 2013.":
            "In the UK, the number of girls named 'Amelia' in 2014 was more than it was in 2013.",
            "What was the number of girls named ‘Amelia’ in 2010 in the UK?": 
            "What was the number of girls named 'Amelia' in 2010 in the UK?",
            "About what was the ratio of the number of girls named Olivia to those named Isla in 2014 in the UK?":
            "About what was the ratio of the number of girls named 'Olivia' to those named 'Isla' in 2014 in the UK?",
            "Which city’s metro system does lie outside the relationship between the total system length and the number of stations most?":
            "Which city's metro system does lie outside the relationship between the total system length and the number of stations most?"
        })

        return data

    def process_ggr(self, original_df):
        data = original_df.replace({
            "What is the difference between the percentage of patients who recovered after a surgery and the percetage of patients who recovered after radiation therapy?'":
            "What is the difference between the percentage of patients who recovered after a surgery and the percentage of patients who recovered after radiation therapy?",
            "Approximately what percentage of people had Adeolitis in the year 2000":
            "Approximately what percentage of people had Adeolitis in the year 2000?",
            "Of 100 patients with disease X, how many are women":
            "Of 100 patients with disease X, how many are women?"
        })
        return data

    def process_holf(self, original_df):
        data = original_df.replace({
            "davinci": "holf"
        })
        return data

    def process_calvi(self, original_df):
        data = original_df.copy()
        def seprate_trick_items(r):
            if r["testType"].find("calvi") != -1:
                if r["imageFile"].find("T") != -1:
                    return "calvi-trick"
                else:
                    return "calvi-standard"
            else:
                return r['testType']

        data['testType'] = data.apply(seprate_trick_items, axis=1)
        return data

    def process_chartqa_test(self, original_df):
        data = original_df.copy()
        questions = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-test-categorical/questions.csv")
        
        def sperate_continuous_categorical(r):
            if r['testType'].find("chartqa") != -1:
                if r['imageFile'] in questions['image_file'].values and r['question'] in questions['question'].values:
                    r['testType'] = "chartqa-test-catgeorical"
                else:
                    r['testType'] = "chartqa-test-continuous"

            return r
        
        return data.apply(sperate_continuous_categorical, axis=1)



if __name__ == "__main__":
    # all_prompts = pd.read_csv("/Users/arnav/Downloads/prompts_apr13.csv", low_memory=False)
    all_prompts = pd.read_csv("/Users/arnav/Downloads/prompts_may26.csv", low_memory=False)
    prompt_types = [
        # "instructions_question", 
        "indist_instructions_question",
        # "incontext_instructions_question"
    ]

    # top_ps = [0.2, 0.4, 0.6, 0.8][::-1]
    # top_ps = [0.4]
    # temperatures = []
    response_count_per_question = 10
    # temperatures = [0.2, 0.4, 0.6, 0.8, 1.0][::-1]
    
    prompt_types = ["indist_instructions_question"]
    # prompt_types = ["instructions_question"]
    # prompt_types = ["indist_instructions_cot_0shot_question"]
    # test_types = ['chartqa-test-categorical']
    test_types = [
        "ggr",
        # "vlat",
        # "holf",
        # 'calvi-trick',
        # 'calvi-standard',
        # 'holf2',
        # 'chartqa-test-continuous',
        # 'chartqa-test-categorical'
    ]
    temperatures = [0.2]
    top_ps = []
    agent_types = [
        # "GPT-4V",
        # "Salesforce/blip2-flan-t5-xxl", 
        # "Salesforce/blip2-flan-t5-xl",
        # "llava-hf/llava-1.5-7b-hf",
        # "Salesforce/blip2-opt-2.7b",
        #  "llava-hf/llava-1.5-13b-hf",
        # "liuhaotian/llava-v1.6-34b",
        # "google/matcha-chartqa",
        # "google/pix2struct-chartqa-base",
        # "Human/Math-2-1",
        # "Human/Math-3",
        "Human"
    ]
    model_response = ModelResponse()
    for prompt_type in prompt_types:
        for test_type in test_types:
            for top_p in top_ps:
                temperature = 1
                for agent_type in agent_types:
                    agent_response_df = model_response.process_model_responses(
                        all_prompts, 
                        agent_type=agent_type,
                        prompt_type=prompt_type,
                        test_type=test_type,
                        temperature=temperature,
                        top_p=top_p,
                        response_count_per_question=response_count_per_question
                    )

                    top_p_dir = 'p' + str(top_p).replace(".", "")
                    temperature_dir= 't' + str(temperature).replace(".", "")
                    # aws_dir = f'responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                    aws_dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                    _dir = f'./{test_type}/{aws_dir}'
                    file_name = f'{_dir}/model_responses.csv'
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)

                    agent_response_df.to_csv(file_name)
                    upload_file_to_s3(file_name, test_type, f'{aws_dir}/model_responses.csv')
                    print("uploaded file")

            for temperature in temperatures:
                top_p = 1
                for agent_type in agent_types:
                    agent_response_df = model_response.process_model_responses(
                        all_prompts, 
                        agent_type=agent_type,
                        prompt_type=prompt_type,
                        test_type=test_type,
                        temperature=temperature,
                        top_p=top_p,
                        response_count_per_question=response_count_per_question
                    )

                    top_p_dir = 'p' + str(top_p).replace(".", "")
                    temperature_dir= 't' + str(temperature).replace(".", "")
                    # aws_dir = f'responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                    aws_dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                    _dir = f'./{test_type}/{aws_dir}'
                    file_name = f'{_dir}/model_responses.csv'
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)

                    print(_dir)
                    agent_response_df.to_csv(file_name)
                    upload_file_to_s3(file_name, test_type, f'{aws_dir}/model_responses.csv')
                    print("uploaded file")

                # df = pd.concat([
                #     model_response.process_model_responses(
                #         all_prompts, 
                #         agent_type=agent_type,
                #         prompt_type=prompt_type,
                #         test_type=test_type,
                #         temperature=temperature,
                #         top_p=top_p,
                #         response_count_per_question=response_count_per_question
                #     ) for agent_type in agent_types])
               
                # top_p_dir = 'p' + str(top_p).replace(".", "")
                # temperature_dir= 't' + str(temperature).replace(".", "")

                # if not os.path.exists(f'./responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}'):
                #     os.makedirs(f'./responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}')

                # file_name = f'./responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}/model_responses.csv'
                # df.to_csv(file_name)
                # upload_file_to_s3(file_name, test_type, f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/model_responses.csv')
                # print("uploaded file")

            
                    # upload_file_to_s3(f'./responses/{prompt_type}/{test_type}_responses.csv', f'{prompt_type}/{test_type}_responses.csv')

    # df = model_response.process_model_responses(
    #     all_prompts, 
    #     agent_type=agent_type,
    #     prompt_type=prompt_type,
    #     test_type=test_type,
    #     temperature=temperature,
    #     top_p=top_p,
    #     response_count_per_question=response_count_per_question
    # )

    # for test_type in datasets:
    #     model_responses = []
    #     for i in range(len(models)):
    #         print(models[i])
            
    #         df = process_model_responses(
    #             all_prompts, 
    #             agent_type=models[i],
    #             prompt_type=prompt_type,
    #             test_type=test_type
    #         )
    #         model_responses.append(df)

    #     model_responses = pd.concat(model_responses)
    #     model_responses = model_responses.apply(add_mc_correctness, axis=1)
    #     if (test_type == "chartqa-test-continuous"):
    #         model_responses = add_continous_error(model_responses)
    #     print(test_type)
    #     model_responses.to_csv(f'./responses/{prompt_type}/{test_type}_responses.csv')