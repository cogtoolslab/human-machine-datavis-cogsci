import altair as alt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator, PCG64
alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()
import os
from evaluation_metrics import EvaluationMetrics

class AccuracyPlot:

    def __init__(
        self, 
        testName, 
        prompt_type, 
        units_of_measure = 'a_in_b', 
        top_p_dir='p04', 
        temperature_dir='t1', 
        model_config=[],
        print_stats=False,
        relax_value = 0.10
    ):
        self.test_name = testName
        self.AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
        self.questions = pd.read_csv(f'{self.AWS_PREFIX}/{testName}/questions.csv')
        self.top_p_dir = top_p_dir
        self.temperature_dir = temperature_dir
        self.evaluation_metric = EvaluationMetrics()
        self.units_of_measure = units_of_measure
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
        raw_model_response_url =  f'{self.AWS_PREFIX}/{testName}/responses/{prompt_type}/{top_p_dir}/{temperature_dir}/model_responses.csv'
        model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{testName}/{_dir}/processed_extracted_responses.csv'
        
        print(model_response_url)
        self.model_responses = pd.read_csv(model_response_url)
        self.model_responses = self.model_responses[self.model_responses["testType"] == self.test_name]
        self.model_responses = self.model_responses.rename({
            "imageFile": "image_file"
        }, axis=1).dropna(subset=["agent_response"])

        self.human_responses = pd.read_csv(raw_model_response_url)
        self.human_responses = self.human_responses[(self.human_responses["agentType"] == "Human/Math-2-1") | 
                                                    (self.human_responses["agentType"] == "Human/Math-3") | 
                                                    (self.human_responses["agentType"] == "Human")]

        self.model_responses = pd.concat([self.model_responses, self.human_responses])
        self.model_responses["correct_answer"] = self.model_responses["correctAnswer"]
        
        self.is_numerical_test = self.test_name in ["chartqa-test-continuous", "holf", "holf2"]

        self.model_responses["agent_response"] = self.model_responses.apply(
            lambda r : self.evaluation_metric.remove_prompt(r["agent_response"], r["prompt"]),
            axis=1
        )

        if print_stats:
            for agent in model_config:
                agent_type = agent[0]
                agent_responses = self.model_responses[self.model_responses["agentType"] == agent_type]
                print(f"Agent {agent_type} only responded to {len(agent_responses) / len(self.questions)} questions for test {self.test_name}")

        if self.units_of_measure == "a_in_b":

            self.model_responses["a_in_b"] = self.model_responses.apply(
                lambda r: self.evaluation_metric.a_in_b(r['correct_answer'], r['agent_response']), axis=1
            )

        # if self.is_numerical_test:
            # self.model_responses["absolute_error"] = self.model_responses.apply(
            #     lambda r: self.evaluation_metric.get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
            # )
            # self.model_responses = self.model_responses.dropna(subset=["absolute_error"])
        
        if self.units_of_measure == "minmax_axis_normalized_error":
            # self.model_responses["agent_response"] = self.model_responses.apply(
            #     lambda r : self.evaluation_metric.get_best_numerical_response(r["agent_response"], r["correct_answer"]),
            #     axis=1
            # ).dropna()
            self.model_responses["agent_response"] = self.model_responses["agent_response"].astype(float)

            if relax_value > 0:
                top_quantile = self.model_responses["agent_response"].quantile(1-relax_value)
                bottom_quantile = self.model_responses['agent_response'].quantile(relax_value)

                def filter_outliers(row):
                    return (row['agent_response'] <= top_quantile) and (row['agent_response'] >= bottom_quantile)

                self.model_responses = self.model_responses[self.model_responses.apply(filter_outliers, axis=1)]

            self.model_responses["error"] = self.model_responses.apply(
                lambda r: self.evaluation_metric.get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
            )
            self.model_responses = self.model_responses.dropna(subset=["error"])

            merged_response = self.questions[["question", "image_file", "min_label", "max_label"]]
            self.model_responses = self.model_responses.merge(merged_response, on=["question", "image_file"])
            
            self.model_responses[self.units_of_measure] = self.model_responses.apply(
                lambda r: self.evaluation_metric.minmax_normalized_error(r['error'], r['min_label'], r['max_label']), axis=1
            )

            

            # self.max_point = 2

            # self.model_responses = self.model_responses[self.model_responses[self.units_of_measure] < self.max_point]

            # p25 = self.model_responses[self.units_of_measure].quantile(0.025)
            # p75 = self.model_responses[self.units_of_measure].quantile(0.975)
            # self.model_responses = self.model_responses[
            #     (self.model_responses[self.units_of_measure] >= p25) & 
            #     (self.model_responses[self.units_of_measure] <= p75)
            # ]

            # self.model_responses[self.units_of_measure] = self.model_responses.apply(
            #     lambda r: self.evaluation_metric.minmax_normalized_error(r['absolute_error'], 0, 100), axis=1
            # )
            self.model_responses = self.model_responses.dropna(subset=["minmax_axis_normalized_error"])
        
        self.model_config = model_config
        # try: 
        #     self.human_responses = pd.read_csv(f'{self.AWS_PREFIX}/{testName}/responses/human_responses.csv')
        #     self.human_responses = self.human_responses.rename({
        #         "imageFile": "image_file"
        #     }, axis=1)
        #     human_agents = self.human_responses["agentType"].unique()
        #     if 'Human/Math-2-1' in human_agents:
        #         self.model_config.append(('Human/Math-2-1', '#639460', '#C8EBC6'))
        #     if 'Human/Math-3' in human_agents:
        #         self.model_config.append(('Human/Math-3', '#2e693b', '#73D287'))
        #     if 'Human' in human_agents:
        #         self.model_config.append(('Human', '#2e693b', '#73D287'))
        # except:

        self.human_responses = None
        

    def bootstrap_ci(
            self, 
            raw_data, 
            n_iterations=10000,
            statistic=np.mean):
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
            bs_mean = statistic(filter_df[self.units_of_measure]) #.mean() # Calculate mean of the filtered data
            return bs_mean
        means = Parallel(n_jobs=-1)(
            delayed(bootstrap_iteration)(df, rng.choice(questions, n_size,  replace=True)) for _ in range(n_iterations)
        )
        
        # 95% confidence interval
        lower = np.percentile(means, 2.5)
        upper = np.percentile(means, 97.5)
        
        return lower, upper


    def create_confidence_interval_df(self, data, statistic=np.mean):
        data_list = []
        num_questions = len(self.questions[["question", "image_file"]])

        for agent in data["agentType"].unique():
            agent_res = data[data["agentType"] == agent]

            lower, upper = self.bootstrap_ci(agent_res, statistic=statistic)

            data_list.append({
                "agentType": agent,
                "ci_upper": upper, 
                "ci_lower": lower,
                "value_count": len(agent_res[['question', 'image_file']].value_counts()) / num_questions,
                "mean": statistic(agent_res[self.units_of_measure])
            })

        return pd.DataFrame(data_list)

    def create_accuracy_df(self, data):
        acc_df = data.groupby(["question", "image_file", "agentType"])[self.units_of_measure].mean()
        return acc_df.reset_index()
    
    def create_error_scatter_plot(self):
        model_order = [c[0] for c in self.model_config]
        color_order = [c[1] for c in self.model_config]

        

        if self.units_of_measure == "absolute_error":
            self.model_responses = self.model_responses[self.model_responses['absolute_error'] <= 1000000]
            self.model_responses = self.model_responses[self.model_responses['absolute_error'] >= 0.001]
            scale=alt.Scale(
                domain=[0, 1000000],
                type="log"
            )
        elif self.units_of_measure == "minmax_axis_normalized_error":
            self.model_responses = self.model_responses[self.model_responses['minmax_axis_normalized_error'] <= 1]
            self.model_responses = self.model_responses[self.model_responses['minmax_axis_normalized_error'] >= 0]
            scale=alt.Scale()
            # domain = [0, 1]
        
        accuracy_df = self.create_accuracy_df(self.model_responses)
        scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=8, opacity=0.7).encode(
            y=alt.Y(f"{self.units_of_measure}:Q", title=f"{self.units_of_measure}", scale=scale), #, 
            x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
            xOffset="jitter:Q",
            color=alt.Color('agentType:N', scale=alt.Scale(domain=model_order, range=color_order)).legend(None),
            ).transform_calculate(
                jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
            )

        return scatter_plot

    def create_error_plot(self, hide_y_title=False, render_ci=True, max_point=False, statistic=np.mean):
        """ Creates a scatter plot with error bars for accuracy data.

        Args:
            test (str): name of the test
        """
        # response_type = unit_of_measurement #if unit_of_measurement == "Proportion Correct" else "err"
        accuracy_df = self.create_accuracy_df(self.model_responses)

        ytitle = self.units_of_measure if not hide_y_title else None
        model_order = [c[0] for c in self.model_config]
        main_color_order = [c[1] for c in self.model_config]
        bg_color_order = [c[2] for c in self.model_config]
        

        if render_ci:
            try:
                ci_df = pd.read_csv(f'./ci_df/{self.test_name}.csv')
            except:
                ci_df = self.create_confidence_interval_df(self.model_responses, statistic=statistic)
                ci_df.to_csv(f'./ci_df/{self.test_name}.csv')

            error_bars = alt.Chart(ci_df).mark_errorbar().encode(
                x=alt.X("agentType:N"),
                y=alt.Y("ci_upper", title=ytitle),
                y2=alt.Y2("ci_lower"),
                strokeWidth=alt.value(2),
                color=alt.Color('agentType', scale=alt.Scale(domain=model_order, range=main_color_order)).legend(None)
            )

            mean_point_plot = alt.Chart(ci_df, title=self.test_name).mark_point(
                size=40, 
                filled=True,
                strokeWidth=1,
                opacity=1
            ).encode(
                y=alt.Y("mean:Q",),
                x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
                color=alt.Color(
                    'agentType:N', 
                    scale=alt.Scale(
                        domain=model_order, 
                        range=main_color_order
                    )
                ).legend(None),
            )

        if max_point and render_ci:
            max_point = alt.Chart(ci_df).mark_line().encode(
                x=alt.X("agentType:N", scale=alt.Scale(domain=model_order)),
                y=alt.Y("value_count"),
                strokeWidth=alt.value(2.5),
                shape=alt.Shape('agentType:N').legend(None),
                color=alt.value("#2b2b2b") #alt.Color('agentType', scale=alt.Scale(domain=order, range=color_arr)).legend(None)
            )

        if self.units_of_measure == "a_in_b":
            y_domain = [0, 1]
        elif self.units_of_measure == "minmax_axis_normalized_error":
            max_point = ci_df["ci_upper"].max()
            max_point = 1.2 if max_point < 1.2 else max_point

            accuracy_df = accuracy_df[accuracy_df[self.units_of_measure] <= max_point]
            y_domain = [0, max_point]
            # y_domain = [0, accuracy_df[self.units_of_measure].max()+1]
        
        scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=8,).encode(
            y=alt.Y(f"{self.units_of_measure}:Q", title=ytitle, scale=alt.Scale(domain=y_domain)),
            x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
            xOffset="jitter:Q",
            color=alt.Color('agentType:N', scale=alt.Scale(domain=model_order, range=bg_color_order)).legend(None),
        ).transform_calculate(
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
        )

        # else:
        #     ci_df = model_ci_df
        #     # scatter_plot = None
        #     accuracy_df = self.create_accuracy_df(self.model_responses, metric=self.units_of_measure)
        #     scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=8, opacity=0.7).encode(
        #         y=alt.Y(f"{self.units_of_measure}:Q", title=ytitle),
        #         x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
        #         xOffset="jitter:Q",
        #         color=alt.Color('agentType:N', scale=alt.Scale(domain=model_order, range=bg_color_order)).legend(None),
        #         ).transform_calculate(
        #             jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
        #     )

        final_plot = scatter_plot
        
        if render_ci:
            final_plot = final_plot + error_bars + mean_point_plot
        
        return final_plot.resolve_scale(color="independent",  opacity="independent")


def create_single_model_comparison_plots(prompt_type="indist_instructions_question", temperature_dir='t1', top_p_dir='p04'):
    test_types = [
        # "ggr",
        "vlat", 
        # "holf",
    ]
    model_sets = [
        ('llava-hf/llava-1.5-7b-hf', '#f58518', '#f9b574'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#8CBEDD'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#6F93B9'),
        ('GPT-4V', '#b85536', '#CD8872'),
        ('Human/Math-2-1', "#639460", "#C8EBC6"),
        ('Human/Math-3', "#2e693b", "#73D287"),
    ]
    test_plots = []
    for test_type in test_types:
        relax_value = 0.00

        accuracy_plot = AccuracyPlot(
            test_type, 
            prompt_type, 
            top_p_dir=top_p_dir, 
            temperature_dir=temperature_dir,
            model_config=model_sets,
            units_of_measure='minmax_axis_normalized_error' if test_type in ["holf", "holf2"] else 'a_in_b',
            print_stats=True,
            relax_value=relax_value
        )
                    
        plot = accuracy_plot.create_error_plot(
            render_ci=True,
            statistic=np.median if test_type in ["holf", "holf2"] else np.mean
        ).properties(title=f"{test_type}", height=300, width=125)
        _dir = f'./single_accuracy_plots/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}'
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        file_name = f"{_dir}/accuracy_plot_med_p00-3.pdf"
        # plot.save(file_name)
        # test_plots.append(plot)

        print(f"Saving plot to {file_name}")
    # test_plots = alt.vconcat(*test_plots).resolve_scale(
    #     x='shared'
    # ).configure_axis(
    #     labels=False
    # )


def create_model_comparison_plots(prompt_type):
    """
    Creates plot with all models across all temperatures and p values
    """
    
    # temperature_dirs =['t10', 't02', 't04' , 't06', 't08']
    temperature_dirs =['t10']
    top_p_dirs = ['p04']
    # top_p_dirs = ['p02', 'p04' , 'p06', 'p08']

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
    metric = 'a_in_b'

    model_sets = [
        [('llava-hf/llava-1.5-7b-hf', '#f9b574', '#f9b574')],
        # ('llava-hf/llava-1.5-13b-hf', '#f58518', '#f58518')],
        [('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#5ba3cf'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#4c78a8')],
        # [("google/pix2struct-chartqa-base", '#b9a7d0', '#b9a7d0'),
        # ("google/matcha-chartqa", '#8b6db2', '#8b6db2')],
        # [('liuhaotian/llava-v1.6-34b', '#F1C232', '#BF9000')],
        [('GPT-4V', '#b85536', '#b85536')],
        [('Human/Math-2-1', "#639460", "#C8EBC6")],
        [('Human/Math-3', "#2e693b", "#73D287")],
        # [('Human', '#b85536', '#b85536')],
    ]

    final_plot_set = []
    for model_set in model_sets:
        v_plots = []
        for test_type in test_types:
            plots = []
            for temperature_dir in temperature_dirs:
                top_p_dir = 'p1'
                dir = f'./accuracy_plots/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}'
                if not os.path.exists(dir):
                        os.makedirs(dir)
                try:
                    calvi_acc = AccuracyPlot(test_type, prompt_type, 
                                            top_p_dir=top_p_dir, 
                                            temperature_dir=temperature_dir,
                                            model_config=model_set,
                                            units_of_measure=metric)
                    
                    plot = calvi_acc.create_error_plot(
                        hide_y_title=True
                    ).properties(title=f"{top_p_dir}_{temperature_dir}", #f"{test_type}_",
                                height=100)

                    plots.append(plot)
                    print(f"Saving plot to {dir}/accuracy_plot.pdf")
                    plot.save(f"{dir}/accuracy_plot.pdf")
                except Exception as e:
                    print(f"Error: {e}")

            for top_p_dir in top_p_dirs:
                temperature_dir = 't1'
                dir = f'./accuracy_plots/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}'
                if not os.path.exists(dir):
                        os.makedirs(dir)
                try:
                    calvi_acc = AccuracyPlot(test_type, 
                                            prompt_type, 
                                            top_p_dir=top_p_dir, 
                                            temperature_dir=temperature_dir,
                                            model_config=model_set,
                                            units_of_measure=metric)

                    plot = calvi_acc.create_error_plot(
                        hide_y_title=True
                    ).properties(title=f"{top_p_dir}_{temperature_dir}", #f"{test_type}_{top_p_dir}_{temperature_dir}", 
                                height=100)

                    plots.append(plot)
                    print(f"Saving plot to {dir}/accuracy_plot.pdf")
                    plot.save(f"{dir}/accuracy_plot.pdf")
                except Exception as e:
                    print(f"Error: {e}")
            
            h_plots = alt.hconcat(*plots, spacing=0).resolve_scale(
                y='shared'
            ).properties(title=f"{test_type}")
            
            v_plots.append(h_plots)
            h_plots.save(f"./accuracy_plots/{prompt_type}/{test_type}/accuracy_plot.pdf")

        v_plots = alt.vconcat(*v_plots).resolve_scale(
            x='shared'
        ).configure_axis(
            labels=False
        )

        v_plots.save(f"./accuracy_plots/{prompt_type}/{model_config[0][0].split("/")[-1]}_accuracy_plot.pdf")
        final_plot_set.append(v_plots)
    
    final_plot = alt.hconcat(*final_plot_set).resolve_scale(
        y='shared'
    ).configure_axis(
        labels=False
    )
    final_plot.save(f"./accuracy_plots/{prompt_type}/relaxed_accuracy_pt_plot.pdf")


def create_log_error_plot(temperature_dir, top_p_dir, prompt_type, units_of_measure='absolute_error',):
    test_types = [
        "holf",
        'holf2',
        'chartqa-test-continuous',
    ]
    model_config = [
        ('llava-hf/llava-1.5-7b-hf', '#f9b574', '#f9b574'),
        ('llava-hf/llava-1.5-13b-hf', '#f58518', '#f58518'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#5ba3cf'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#4c78a8'),
        ('GPT-4V', '#b85536', '#b85536'),
        ('liuhaotian/llava-v1.6-34b', '#F1C232', '#BF9000'),
        ("google/pix2struct-chartqa-base", '#b9a7d0', '#b9a7d0'),
        ("google/matcha-chartqa", '#8b6db2', '#8b6db2'),
    ]

    charts = []
    for test_type in test_types:
        try:
            accuracy_plot = AccuracyPlot(test_type, prompt_type, units_of_measure, top_p_dir, temperature_dir, model_config)
            plot = accuracy_plot.create_error_scatter_plot()
            charts.append(plot)
        except Exception as e:
            print(f"Error: {e}")

    h_charts = alt.hconcat(*charts).properties(title=f"{prompt_type} + {top_p_dir} + {temperature_dir}")

    dir = f'./accuracy_plots/{units_of_measure}/{prompt_type}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(f"Saving plot to {dir}/log_error_plot.pdf")
    h_charts.save(f"{dir}/log_error_plot.pdf")


def create_axis_minmax_error_plot(temperature_dir, top_p_dir, prompt_type, units_of_measure='minmax_axis_normalized_error',):
    test_types = [
        "holf",
        'holf2',
    ]

    model_config = [
        ('llava-hf/llava-1.5-7b-hf', '#f9b574', '#f9b574'),
        ('llava-hf/llava-1.5-13b-hf', '#f58518', '#f58518'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#5ba3cf'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#4c78a8'),
        ('GPT-4V', '#b85536', '#b85536'),
        ('liuhaotian/llava-v1.6-34b', '#F1C232', '#BF9000'),
        ("google/pix2struct-chartqa-base", '#b9a7d0', '#b9a7d0'),
        ("google/matcha-chartqa", '#8b6db2', '#8b6db2'),
    ]

    charts = []
    for test_type in test_types:
        try:
            accuracy_plot = AccuracyPlot(test_type, prompt_type, units_of_measure, top_p_dir, temperature_dir, model_config)
            plot = accuracy_plot.create_error_scatter_plot()
            charts.append(plot)
        except Exception as e:
            print(f"Error: {e}")

    h_charts = alt.hconcat(*charts).properties(title=f"{prompt_type} + {top_p_dir} + {temperature_dir} ")

    dir = f'./accuracy_plots/{units_of_measure}/{prompt_type}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(f"Saving plot to {dir}/error_plot.pdf")
    h_charts.save(f"{dir}/error_plot.pdf")
    
if __name__ == "__main__":
    # prompt_type = 'indist_instructions_question'
    prompt_type = 'indist_instructions_cot_0shot_question'
    metric = 'relaxed_accuracy'
   
    test_types = [
        "ggr",
        "vlat", 
        "holf",
        # "Human/Math-2-1",
        # "Human/Math-3",
        # "Human"
        # 'calvi-trick',
        # 'calvi-standard',
        # 'holf2',
        # 'chartqa-test-continuous',
        # 'chartqa-test-categorical'
    ]
    temperature_dirs =['t10', 't02', 't04' , 't06', 't08']
    top_p_dirs = ['p02', 'p04' , 'p06', 'p08']
    top_p = 'p1'
    top_t = 't10'

    create_single_model_comparison_plots()
    # create_model_comparison_plots("indist_instructions_cot_0shot_question")
    # create_log_error_plot(top_t, top_p, prompt_type, units_of_measure='absolute_error')
    # create_axis_minmax_error_plot(top_t, top_p, prompt_type, units_of_measure='minmax_axis_normalized_error')
    