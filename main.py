from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


# New block of general instructions to be prepended to all prompts
GENERAL_INSTRUCTIONS = clean_indents(
    """
    **General Operating Principles:**
    Your response must adhere to the following rules.

    **Style and Tone:**
    - Use British English and grammar
    - Be as terse as possible
    - No em dashes, Oxford commas, or full stops at the end of list items
    - No extra bolding or italicising
    - No formalities, praise, exclamation marks, or emojis
    - Write naturally, not like a report. Avoid repetition and disclaimers (e.g., "I am an AI", "I am not an expert")

    **Reasoning and Analysis:**
    - Quantify statements and use explicit probabilities whenever possible
    - Your analysis should be robust enough to withstand critique from economists and superforecasters
    - Consider counterfactuals and acknowledge trade-offs
    - Explicitly notice and explain your uncertainties and confidence levels
    - Be sceptical of all information, including your own. Flag unverified claims with [may need verification]
    - When presented with statistics, polls, or research, first critique the methodology, sampling, and potential for bias before accepting the premise
    - Flag any assumptions you make to fill in gaps not explicitly stated in the question
    - Provide context (e.g., geography, timeframe) for all quantitative or trend-based statements
    - Do not state obvious facts; assume a strong user knowledge base in economics and analytics
    - Prioritise the most important hypotheses; do not give equal weight to all ideas

    **Language Constraints:**
    - Taboo words (and their variants): "fundamental", "engaging", "deep", "captivating", "dance", "crucial"
    - Avoid formulaic sentences like "isn't just x, but also y" or platitudes
    - Preferred terminology for countries: LMICs, HICs, richer/poorer countries

    **Interaction Protocol:**
    - Do not provide sycophantic flattery or hollow validation
    - Probe assumptions, surface bias, present counter-evidence, challenge emotional framing, and disagree openly when warranted
    """
)

SCEPTIC_PROMPT_BINARY = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- Sceptic's Task ---
    You are a 'Base Rate Absolutist', a forecaster whose predictions are rigorously anchored to historical precedent.

    Your Task: A High-Impact analyst has presented a case for a particular outcome. Your job is to first establish the historical base rate and then critically evaluate if the analyst's argument is strong enough to justify deviating from it.

    Your Process:
    1.  **Determine the Base Rate:** From the provided research on "{{question_text}}", what is the historical base rate for events of this type? State this clearly
    2.  **Anchor on the Base Rate:** This number is your anchor. Your default forecast must be very close to it
    3.  **Evaluate the Counter-Argument:** The High-Impact Analyst has made this case: "{{counter_argument}}". Does this argument provide *extraordinary, quantifiable evidence* of a shift that makes the historical base rate irrelevant? Mere narrative or speculation is insufficient
    4.  **Produce Final Forecast:** Based on your analysis, write a single, dense paragraph. It must begin by stating the base rate, then explain why you are either sticking to it or deviating slightly based on the strength (or weakness) of the counter-argument

    **Time Decay Analysis:** {{time_decay_info}}
    **Full Research Text:**
    {{research}}

    The last thing you write must be your final answer as: "Probability: ZZ%"
    """
)

HIGH_IMPACT_PROMPT_BINARY = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- High-Impact Analyst's Task ---
    You are a High-Impact Analyst, skilled at identifying unlikely but plausible "black swan" events.
    Your task is to find the single most compelling chain of events from the provided research that would cause this event **to happen**.
    Acknowledge that this path is unlikely, but argue for its plausibility.
    Your entire rationale must be a single, dense paragraph.

    Research on "{{question_text}}":
    {{research}}

    The last thing you write must be your final answer as: "Probability: ZZ%"
    """
)

SCEPTIC_PROMPT_NUMERIC = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- Sceptic's Task ---
    You are a 'Base Rate Absolutist', a forecaster who anchors on historical trends for numeric questions.
    A High-Impact analyst has provided a wide, volatile distribution. Your task is to rebut it with a more conservative forecast anchored in the data.

    Your Process:
    1.  **Determine the Base Rate / Trend:** From the research on "{{question_text}}", what is the historical trend, average, or status quo value? This is your central anchor (50th percentile)
    2.  **Evaluate the Counter-Argument:** The High-Impact Analyst's reasoning is: "{{counter_argument}}". Does this justify their extreme distribution, or is it speculative?
    3.  **Produce Final Forecast:** Provide a tight, conservative distribution of percentiles anchored around the base rate. Your rationale should be a single, dense paragraph explaining why a deviation from the historical trend is unlikely

    **Time Decay Analysis:** {{time_decay_info}}
    **Full Research Text:**
    {{research}}

    The last thing you write is your final answer with percentile estimates from 10 to 90.
    """
)

HIGH_IMPACT_PROMPT_NUMERIC = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- High-Impact Analyst's Task ---
    You are a High-Impact Analyst, skilled at identifying "fat tail" risks for numeric questions.
    Your task is to produce a wide distribution that accounts for plausible black swan events, based on the research for "{{question_text}}".
    Your rationale should be a single, dense paragraph arguing why a volatile or unexpected outcome is more likely than the boring trendline.
    
    Full Research Text:
    {{research}}
    
    The last thing you write is your final answer with percentile estimates from 10 to 90.
    """
)

SCEPTIC_PROMPT_MC = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- Sceptic's Task ---
    You are a 'Base Rate Absolutist', a forecaster who anchors on the most likely or status quo option in a multiple-choice question.
    A High-Impact analyst has argued for an unlikely option. Your task is to rebut their case and provide a conservative forecast anchored on the most probable outcome.

    Your Process:
    1.  **Determine the Base Rate / Favourite:** From the research on "{{question_text}}", which of the options {{options}} is the clear favourite or status quo?
    2.  **Evaluate the Counter-Argument:** The High-Impact Analyst's reasoning is: "{{counter_argument}}". Does this justify assigning high probability to a long-shot option?
    3.  **Produce Final Forecast:** Provide probabilities for all options that heavily favour the most likely outcome. Your rationale should be a single, dense paragraph explaining why the status quo is likely to hold

    **Full Research Text:**
    {{research}}
    
    The last thing you write is your final probabilities for all options.
    """
)

HIGH_IMPACT_PROMPT_MC = clean_indents(
    f"""
    {GENERAL_INSTRUCTIONS}

    --- High-Impact Analyst's Task ---
    You are a High-Impact Analyst, skilled at identifying plausible "dark horse" candidates in multiple-choice questions.
    Your task is to make the strongest case for an unlikely option being more probable than commonly thought, based on the research for "{{question_text}}".
    The options are: {{options}}.
    Your rationale should be a single, dense paragraph. Your probability distribution should reflect this contrarian view by assigning a surprisingly high probability to one of the non-favourite options.
    
    Full Research Text:
    {{research}}

    The last thing you write is your final probabilities for all options.
    """
)


# We now define our "Team of Rivals" with all the prompts
MODEL_CROWD = [
    {
        "persona": "Sceptic", 
        "model": "openrouter/openai/gpt-4o", 
        "prompts": {
            "binary": SCEPTIC_PROMPT_BINARY, 
            "numeric": SCEPTIC_PROMPT_NUMERIC, 
            "multiple_choice": SCEPTIC_PROMPT_MC
        }
    },
    {
        "persona": "High-Impact", 
        "model": "openrouter/anthropic/claude-3-5-sonnet", 
        "prompts": {
            "binary": HIGH_IMPACT_PROMPT_BINARY,
            "numeric": HIGH_IMPACT_PROMPT_NUMERIC,
            "multiple_choice": HIGH_IMPACT_PROMPT_MC
        }
    },
]

class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    

    Main changes since Q2:

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.

    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Runs research using Anthropic API for analysis
        """
        logger.info(f"Running research for '{question.question_text}' using Anthropic...")
    
        try:
            research_model = GeneralLlm(
                model="anthropic/claude-sonnet-4-20250514",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3
            )
            
            research_prompt = f"""
            Analyse this forecasting question: {question.question_text}
            
            Provide comprehensive analysis covering:
            - Historical context and relevant precedents
            - Key factors that influence such outcomes
            - Base rates for similar events
            - Current trends and patterns (within your knowledge)
            - Important considerations for forecasting
            - Potential scenarios and their likelihood drivers
            
            Focus on factual analysis that would inform an accurate probability estimate.
            Be explicit about knowledge limitations and uncertainty.
            """
            
            research_text = await research_model.invoke(research_prompt)
            logger.info("Successfully completed research analysis")
            return research_text
        
        except Exception as e:
            logger.error(f"Anthropic research failed: {e}")
            return f"Research analysis failed: {e}"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        
        high_impact_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "High-Impact"), None)
        sceptic_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "Sceptic"), None)

        if not high_impact_analyst or not sceptic_analyst:
            raise ValueError("Both Sceptic and High-Impact personas must be in MODEL_CROWD")

        try:
            logger.info("Running High-Impact Analyst for binary question...")
            high_impact_prompt = high_impact_analyst["prompts"]["binary"].format(
                question_text=question.question_text, research=research
            )
            litellm_params = {}
            if "openai" in high_impact_analyst["model"]:
                litellm_params = {"service_tier": "flex"}
            high_impact_llm = GeneralLlm(model=high_impact_analyst["model"], litellm_params=litellm_params)
            counter_argument = await high_impact_llm.invoke(high_impact_prompt)
        except Exception as e:
            logger.error(f"High-Impact Analyst failed: {e}")
            counter_argument = "The High-Impact Analyst failed to produce a counter-argument."

        now = datetime.now(timezone.utc)
        total_duration = (question.close_time - question.published_time).total_seconds()
        time_remaining = (question.close_time - now).total_seconds()
        
        time_decay_info = "Not a time-sensitive question."
        if total_duration > 3600 and time_remaining > 0:
            time_elapsed_percent = (1 - (time_remaining / total_duration)) * 100
            days_remaining = time_remaining / (24 * 3600)
            time_decay_info = (
                f"The forecasting period is {time_elapsed_percent:.0f}% complete with {days_remaining:.1f} days remaining. "
                "The event has not yet occurred. The probability should be decayed accordingly."
            )
        
        logger.info("Running Sceptic Analyst to rebut and provide final forecast...")
        sceptic_prompt_text = sceptic_analyst["prompts"]["binary"].format(
            question_text=question.question_text,
            research=research,
            time_decay_info=time_decay_info,
            counter_argument=counter_argument
        )
        
        litellm_params = {}
        if "openai" in sceptic_analyst["model"]:
            litellm_params = {"service_tier": "flex"}
        sceptic_llm = GeneralLlm(model=sceptic_analyst["model"], litellm_params=litellm_params)
        final_reasoning = await sceptic_llm.invoke(sceptic_prompt_text)

        binary_prediction: BinaryPrediction = await structure_output(
            final_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        final_prediction = max(0.005, min(0.995, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Final Answer taken from Sceptic persona after rebuttal: {final_prediction*100:.1f}%"
        )
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=final_reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        
        high_impact_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "High-Impact"), None)
        sceptic_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "Sceptic"), None)

        if not high_impact_analyst or not sceptic_analyst:
            raise ValueError("Both Sceptic and High-Impact personas must be in MODEL_CROWD")

        try:
            logger.info("Running High-Impact Analyst for numeric question...")
            high_impact_prompt = high_impact_analyst["prompts"]["numeric"].format(
                question_text=question.question_text, research=research
            )
            litellm_params = {}
            if "openai" in high_impact_analyst["model"]:
                litellm_params = {"service_tier": "flex"}
            high_impact_llm = GeneralLlm(model=high_impact_analyst["model"], litellm_params=litellm_params)
            counter_argument = await high_impact_llm.invoke(high_impact_prompt)
        except Exception as e:
            logger.error(f"High-Impact Analyst failed: {e}")
            counter_argument = "The High-Impact Analyst failed to produce a counter-argument."

        now = datetime.now(timezone.utc)
        total_duration = (question.close_time - question.published_time).total_seconds()
        time_remaining = (question.close_time - now).total_seconds()
        time_decay_info = "Not a time-sensitive question."
        if total_duration > 3600 and time_remaining > 0:
            time_elapsed_percent = (1 - (time_remaining / total_duration)) * 100
            days_remaining = time_remaining / (24 * 3600)
            time_decay_info = f"The forecasting period is {time_elapsed_percent:.0f}% complete."

        logger.info("Running Sceptic Analyst to rebut and provide final forecast...")
        sceptic_prompt_text = sceptic_analyst["prompts"]["numeric"].format(
            question_text=question.question_text,
            research=research,
            time_decay_info=time_decay_info,
            counter_argument=counter_argument
        )
        
        litellm_params = {}
        if "openai" in sceptic_analyst["model"]:
            litellm_params = {"service_tier": "flex"}
        sceptic_llm = GeneralLlm(model=sceptic_analyst["model"], litellm_params=litellm_params)
        final_reasoning = await sceptic_llm.invoke(sceptic_prompt_text)

        percentile_list: list[Percentile] = await structure_output(
            final_reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        final_prediction = NumericDistribution.from_question(percentile_list, question)
        
        logger.info(
            f"Final Answer taken from Sceptic persona after rebuttal: {final_prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=final_reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        
        high_impact_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "High-Impact"), None)
        sceptic_analyst = next((analyst for analyst in MODEL_CROWD if analyst["persona"] == "Sceptic"), None)

        if not high_impact_analyst or not sceptic_analyst:
            raise ValueError("Both Sceptic and High-Impact personas must be in MODEL_CROWD")

        try:
            logger.info("Running High-Impact Analyst for multiple choice question...")
            high_impact_prompt = high_impact_analyst["prompts"]["multiple_choice"].format(
                question_text=question.question_text, research=research, options=question.options
            )
            litellm_params = {}
            if "openai" in high_impact_analyst["model"]:
                litellm_params = {"service_tier": "flex"}
            high_impact_llm = GeneralLlm(model=high_impact_analyst["model"], litellm_params=litellm_params)
            counter_argument = await high_impact_llm.invoke(high_impact_prompt)
        except Exception as e:
            logger.error(f"High-Impact Analyst failed: {e}")
            counter_argument = "The High-Impact Analyst failed to produce a counter-argument."
        
        logger.info("Running Sceptic Analyst to rebut and provide final forecast...")
        sceptic_prompt_text = sceptic_analyst["prompts"]["multiple_choice"].format(
            question_text=question.question_text,
            research=research,
            options=question.options,
            counter_argument=counter_argument
        )
        
        litellm_params = {}
        if "openai" in sceptic_analyst["model"]:
            litellm_params = {"service_tier": "flex"}
        sceptic_llm = GeneralLlm(model=sceptic_analyst["model"], litellm_params=litellm_params)
        final_reasoning = await sceptic_llm.invoke(sceptic_prompt_text)

        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=final_reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
        )
        logger.info(
            f"Final Answer taken from Sceptic persona after rebuttal: {predicted_option_list}"
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=final_reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        # The loop is now handled inside the bot, so this should be 1
        predictions_per_research_report=1,
        
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        
        # Configure the llms dictionary to use the best tools
        llms={
             "default": "openrouter/openai/gpt-4o",
            "parser": "openrouter/openai/gpt-4o-mini",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
           ## "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
           ##  "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
           ## "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)