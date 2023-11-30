# Nepali-Text-Summarization

### About the code
This repository contains code for generating summary of Nepali text from scratch using Tfidf method.

### Requirements:
- nltk

### Code review:
Inside the 'Nepali_Summarizer' folder,
- **summarizer.py**: This script conatins the dataclass which can be used for generating summaries of Nepali text.

### How to use this code:

First, clone the repository:

```bash
!git clone https://github.com/SujalNeupane9/Nepali-Text-Summarization.git
```

Navigate to the folder as:
```bash
%cd Nepali-Text-Summarization
```

Then, in your Jupyter Notebook:

```bash
from Nepali_Summarizer.summarizer import Nepali_summarizer

# Your Nepali text data
text = [
    "नेपाल एक अद्भुत देश हो।",
    # ... (other Nepali sentences)
]

# Create an instance of Nepali_summarizer
summarizer = Nepali_summarizer()

# Generate summary
summary = summarizer.generate_summary(text)

# Print the generated summary
print(summary)
```

This code demonstrates how to use the Nepali_summarizer class to generate a summary from Nepali text data in a Jupyter Notebook environment.
