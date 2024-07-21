from tacl_analysis.evaluation_metrics import EvaluationMetrics
evaluation_metric = EvaluationMetrics()
from numpy.random import Generator, PCG64
from joblib import Parallel, delayed

ggr_questions = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/questions.csv")
vlat_questions = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/questions.csv")
holf_questions = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/questions.csv")

ggr = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv")
holf = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv")
vlat = pd.read_csv('https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/processed_extracted_responses.csv')

ggr_human = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/ggr/responses/indist_instructions_question/p04/t1/model_responses.csv")
ggr_human = ggr_human[(ggr_human["agentType"] == "Human/Math-2-1") | 
                        (ggr_human["agentType"] == "Human/Math-3") | 
                        (ggr_human["agentType"] == "Human")]
vlat_human = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/vlat/responses/indist_instructions_question/p04/t1/model_responses.csv")
vlat_human = vlat_human[(vlat_human["agentType"] == "Human/Math-2-1") | 
                        (vlat_human["agentType"] == "Human/Math-3") | 
                        (vlat_human["agentType"] == "Human")]
holf_human = pd.read_csv("https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/holf/responses/indist_instructions_question/p04/t1/model_responses.csv")
holf_human = holf_human[(holf_human["agentType"] == "Human/Math-2-1") | 
                        (holf_human["agentType"] == "Human/Math-3") | 
                        (holf_human["agentType"] == "Human")]

ggr = pd.concat([ggr, ggr_human])
vlat = pd.concat([vlat, vlat_human])
holf = pd.concat([holf, holf_human])


ggr = ggr[ggr["testType"] == "ggr"]
vlat = vlat[vlat["testType"] == "vlat"]
holf = holf[holf["testType"] == "holf"]

ggr["correct_answer"] = ggr["correctAnswer"]
ggr["a_in_b"] = ggr.apply(
    lambda r: int(evaluation_metric.a_in_b(r['correct_answer'], r['agent_response'])), axis=1
)

vlat["correct_answer"] = vlat["correctAnswer"]
vlat["a_in_b"] = vlat.apply(
    lambda r: int(evaluation_metric.a_in_b(r['correct_answer'], r['agent_response'])), axis=1
)

holf = holf.drop(columns=["min_label", "max_label"])
holf["correct_answer"] = holf["correctAnswer"]
holf["agent_response"] = holf["agent_response"].astype(float)
holf["error"] = holf.apply(
    lambda r: evaluation_metric.get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
)
holf = holf.dropna(subset=["error"])
merged_response = holf_questions[["question", "image_file", "min_label", "max_label"]]
holf = holf.merge(merged_response, on=["question", "image_file"])

holf["minmax_axis_normalized_error"] = holf.apply(
    lambda r: evaluation_metric.minmax_normalized_error(r['error'], r['min_label'], r['max_label']), axis=1
)

def bootstrap_ci(
        raw_data, 
        n_iterations=1000,
        statistic=np.mean,
        units_of_measure="a_in_b"):
    data = raw_data.copy()
    data["question_image"] = data["question"] + " & " + data["image_file"]
    rng = Generator(PCG64())
    questions = list(data["question_image"].unique())
    n_size = len(questions)
    df = data.copy()

    # sample within the data
    df = df.sample(frac=1, replace=True, random_state=1)

    def bootstrap_iteration(data, chosen_qs):
        filter_df = data[data["question_image"].isin(chosen_qs)] # Filter based on chosen questions
        bs_mean = statistic(filter_df[units_of_measure]) #.mean() # Calculate mean of the filtered data
        return bs_mean
    means = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(df, rng.choice(questions, n_size,  replace=True)) for _ in range(n_iterations)
    )
    
    # 95% confidence interval
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return lower, upper


def create_confidence_interval_df(data, statistic=np.mean, units_of_measure="a_in_b"):
    data_list = []
    # num_questions = len(self.questions[["question", "image_file"]])

    for agent in data["agentType"].unique():
        agent_res = data[data["agentType"] == agent]

        lower, upper = bootstrap_ci(agent_res, statistic=statistic, units_of_measure=units_of_measure)

        data_list.append({
            "agentType": agent,
            "ci_upper": upper, 
            "ci_lower": lower,
            # "value_count": len(agent_res[['question', 'image_file']].value_counts()) / num_questions,
            "mean": statistic(agent_res[units_of_measure])
        })

    return pd.DataFrame(data_list)