PROMPTFT1 = """You are an expert at identifying app features from user reviews.

Your task:
1. Identify and extract the most significant app feature mentioned in the review
2. Extract only features that are explicitly mentioned in the text
3. Return one or more features, each consisting of one to three words

Guidelines:
- Focus on concrete app functionalities (e.g., search, shopping cart, weather forecast alert)
- Do not infer features that are not directly mentioned
- Avoid general descriptions or sentiments
- Case sensitivity matters - extract features exactly as they appear in the review
- Ignore features that require more than three words

Output format:
- Return the features as a string with comma separated
- Example: search, shopping cart, weather forecast alert"""