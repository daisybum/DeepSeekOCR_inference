from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

class PromptManager:
    @staticmethod
    def get_location_analysis_prompt():
        """
        Returns a LangChain FewShotPromptTemplate for commercial location analysis.
        Designed for DeepSeek-OCR model.
        """
        
        # 1. Expert Examples (Few-Shot)
        # These examples guide the model on the expected depth and structure of the analysis.
        examples = [
            {
                "instruction": "Analyze the commercial potential of the location shown in the image.",
                "analysis": """<analysis>
1. **Accessibility & Traffic**:
   - The site is located at a busy 4-way intersection with clearly marked crosswalks.
   - A bus stop is visible directly in front of the building, ensuring a steady flow of foot traffic.
   - No immediate subway entrance is visible in the frame, but the road width suggests a main arterial route.

2. **Visibility & Exposure**:
   - The building features a wide glass facade, offering excellent visibility for signage.
   - It is a corner lot, maximizing exposure to traffic from two directions.
   - There are no large trees or structures blocking the view of the storefront.

3. **Surrounding Commercial Environment**:
   - Neighboring tenants include a bank and a pharmacy, which typically draw regular daytime traffic.
   - The area appears to be a mix of office and retail, suggesting a lunch-hour peak.

4. **Strategic Recommendation**:
   - **High Potential**: Coffee shop, Bakery, or Convenience Store.
   - **Reasoning**: The high foot traffic from the bus stop and the corner visibility make it ideal for businesses relying on impulse purchases or convenience.
</analysis>"""
            }
        ]

        # 2. Template for each example
        example_prompt = PromptTemplate(
            input_variables=["instruction", "analysis"],
            template="User: {instruction}\nAssistant: {analysis}"
        )

        # 3. Final Few-Shot Prompt Assembly
        # We prepend <image> and <|grounding|> as required by the DeepSeek-OCR model architecture.
        # We use a clear system-like instruction in the prefix.
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="<image>\n<|grounding|>You are an expert commercial real estate analyst. Analyze the location shown in the image, focusing on accessibility, visibility, competition, and strategic recommendations. Provide your response in the structured format shown below.",
            suffix="User: {user_input}\nAssistant:",
            input_variables=["user_input"]
        )
        
        return prompt
