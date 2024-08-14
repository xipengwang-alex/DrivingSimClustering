import numpy as np

choices_map = {
    'Q130': {
        'I never use this feature': 0,
        'I use this feature sometimes': 1,
        'I use this feature as often as possible': 2,
        "I've never driven a car with this feature": np.nan,
        'I would never use this feature': 0,
        'I would use this feature sometimes': 1,
        'I would use this feature as often as possible': 2
    },
    'Q134': {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Somewhat disagree': 3,
        'Neither agree nor disagree': 4,
        'Somewhat agree': 5,
        'Agree': 6,
        'Strongly agree': 7
    },
    'Q135': {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Somewhat disagree': 3,
        'Neither agree nor disagree': 4,
        'Somewhat agree': 5,
        'Agree': 6,
        'Strongly agree': 7
    },
    'Q136': {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Somewhat disagree': 3,
        'Neither agree nor disagree': 4,
        'Somewhat agree': 5,
        'Agree': 6,
        'Strongly agree': 7
    },
    'Q137': {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Somewhat disagree': 3,
        'Neither agree nor disagree': 4,
        'Somewhat agree': 5,
        'Agree': 6,
        'Strongly agree': 7
    },
    'Q138': {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Somewhat disagree': 3,
        'Neither agree nor disagree': 4,
        'Somewhat agree': 5,
        'Agree': 6,
        'Strongly agree': 7
    },
    'Q110': {
        'None': 0,
        'Slight': 1,
        'Moderate': 2,
        'Severe': 3
    },
    'Q22': {
        'Less than 5,000 miles': 1,
        '5,000-10,000 miles': 2,
        '10,000-15,000 miles': 3,
        '15,000-20,000 miles': 4,
        'More than 20,000 miles': 5
    },
    'Q24': {
        'Daily': 4,
        'A few times a week': 3,
        'A few times a month': 2,
        'A few times a year': 1,
        'Never': 0
    },
    'Q25': {
        'Daily': 4,
        'A few times a week': 3,
        'A few times a month': 2,
        'A few times a year': 1,
        'Never': 0
    },
    'Q26': {
        'Very defensive': 1,
        'Defensive': 2,
        'Neutral': 3,
        'Aggressive': 4,
        'Very Aggressive': 5
    }
}