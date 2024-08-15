# python -3.11
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from choice_map import choices_map 

def parse_and_preprocess_surveys(survey1_path: str, survey2_path: Optional[str] = None, choices_map: Dict[str, Dict[str, int]] = choices_map, output_path: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], pd.DataFrame]:
    df1 = pd.read_csv(survey1_path, skiprows=[1])
    df2 = pd.read_csv(survey2_path, skiprows=[1]) if survey2_path else None

    raw_responses = {}

    surveys = [(df1, 'survey1')]
    if df2 is not None:
        surveys.append((df2, 'survey2'))

    for df, survey_key in surveys:
        for _, row in df.iterrows():
            participant_id = str(row['Q126'])
            if participant_id not in raw_responses:
                raw_responses[participant_id] = {survey_key: {}} if df2 is None else {'survey1': {}, 'survey2': {}}
            
            for column in df.columns:
                if column != 'Q126':
                    raw_responses[participant_id][survey_key][column] = str(row[column])

    encoded_data = []

    for participant_id, surveys in raw_responses.items():
        participant_encoded = {'participant_id': participant_id}
        for survey_key, responses in surveys.items():
            for question, answer in responses.items():
                question_base = question.split('_')[0]
                if question_base in choices_map:
                    encoded_value = choices_map[question_base].get(answer, np.nan)
                    participant_encoded[f"{survey_key}_{question}"] = encoded_value
                else:
                    participant_encoded[f"{survey_key}_{question}"] = answer
        encoded_data.append(participant_encoded)

    encoded_df = pd.DataFrame(encoded_data)

    if output_path:
        encoded_df.to_csv(output_path, index=False)

    return raw_responses, encoded_df


survey1_path = 'Data/Questionnaire 1.csv'
survey2_path = 'Data/Questionnaire 2.csv'
output_path = 'Data/encoded_survey_data.csv' 
#raw_responses, encoded_df = parse_and_preprocess_surveys(survey1_path, survey2_path, choices_map, output_path)

raw_responses, encoded_df = parse_and_preprocess_surveys(survey1_path, choices_map = choices_map, output_path = output_path)

print(encoded_df.head())

pd.set_option('display.max_columns', None)
print(encoded_df)