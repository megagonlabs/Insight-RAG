PROMPTS = {"insight_identifier": ''''You are given a question or task along with its required input. Your goal is to extract the necessary insight that will allow another autoregressive LLM—pretrained on a dataset of scientific papers—to complete the answer. The insight must be expressed as a sentence fragment (i.e., a sentence that is meant to be completed).

Instructions:

Extract the Insight:
Identify the key information needed from the dataset to solve the task or answer the question.
Format the insight as a sentence fragment that can be completed by the LLM trained on the dataset.
For example, if the task is to find the birthplace of Person X, your insight should be:
"Person X was born in".

Determine Answer Multiplicity:
Determine whether the answer should be singular or plural based solely on the plurality of the nouns in the question. Do not use common sense or external context—rely exclusively on grammatical cues in the question. 
For instance, if the question uses plural nouns (e.g., "What are the cities in California?"), set Multi-answer to True. Conversely, if the question uses singular nouns (e.g., "What does pizza contain?"), set it to False.

Relevance Check:
Only include insights that are directly answerable from the dataset.
If an insight does not relate to the available dataset, ignore it.

Output Format:
Return the result as a list of dictionaries.
Each dictionary must have two keys:
"Insight": The sentence fragment containing the key insight.
"Multi-answer": A Boolean (True or False) indicating whether multiple answers are required.
Example Output for follwing questions, Where was Person X born in? what does pizza contain? What are the Cities in California?:

[
  {{"Insight": "Person X was born in", "Multi-answer": false}},
  {{"Insight": "Pizza contains", "Multi-answer": false}},
  {{"Insight": "The cities in California are", "Multi-answer": true}}
]

Please provide your final answer in this JSON-like list-of-dictionaries format with no additional commentary.
Also, make sure to NOT add any extra word to the insights other than the word present in the input. 
Remove all unnecessary words and provide the insight in its simplest form. For example, if the query asks "what are the components that X uses?", the insight should be "X uses". Similarly, if the query asks "what are all the components/techniques/features/applications/outcomes included in Z?", the insight should be "Z include".
If a non-question task is given, possible insights might involve asking about how two concepts are connected or a definition of a concept. Only identify the insight you believe will help solve the task, and provide it as a short sentence fragment to be completed. Do not add any unnecessary content or summaries of the input.
Additionally, for non-question tasks, the insight should NOT refer to the specific input or include any input-specific identifiers (such as Paper-A or Paper-B). Instead, it should be a STAND-ALONE statement focusing on the underlying concepts, entities, and their relationships from the inputs. If you cannot find any such insights, return a list of EMPTY dictionary.

Task:
{}''',
		   "augmented_QA": '''Answer the question using the context. Do not include any extra explanation.
Question: {}
Context: {}''',
		   "augmented_matching": '''You are provided with two research papers, Paper-A and Paper-B, and some useful insights. Your task is to determine if the papers are relevant enough to be cited by the other. You may use the insights to better predict whether the papers are relevant or not. The insights should only serve as supportive evidence; do not rely on them blindly.
Your response must be provided in a JSON format with two keys:
"explanation": A detailed explanation of your reasoning and analysis.
"answer": The final determination ("Yes" or "No").

{}

Useful insights:
{}
'''
}
