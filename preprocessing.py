# python -3.11
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def parse_and_preprocess_surveys(survey1_path: str, survey2_path: str, choices_map: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], pd.DataFrame]:
    df1 = pd.read_csv(survey1_path)
    df2 = pd.read_csv(survey2_path)

    raw_responses = {}

    for df, survey_key in [(df1, 'survey1'), (df2, 'survey2')]:
        for _, row in df.iterrows():
            participant_id = str(row['Q126'])
            if participant_id not in raw_responses:
                raw_responses[participant_id] = {'survey1': {}, 'survey2': {}}
            
            for column in df.columns:
                if column != 'Q126':
                    raw_responses[participant_id][survey_key][column] = str(row[column])

    encoded_data = []

    # encode responses
    for participant_id, surveys in raw_responses.items():
        participant_encoded = {'participant_id': participant_id}
        for survey_key, responses in surveys.items():
            for question, answer in responses.items():
                if question in choices_map:
                    encoded_value = choices_map[question].get(answer, np.nan)
                    participant_encoded[f"{survey_key}_{question}"] = encoded_value
                else:
                    participant_encoded[f"{survey_key}_{question}"] = answer
        encoded_data.append(participant_encoded)

    encoded_df = pd.DataFrame(encoded_data)

    return raw_responses, encoded_df

choices_map = {

}


survey1_path = 'Data/Questionnaire 1.csv'
survey2_path = 'Data/Questionnaire 2.csv'
raw_responses, encoded_df = parse_and_preprocess_surveys(survey1_path, survey2_path, choices_map)

print(encoded_df.head())


print(encoded_df)
'''
'''