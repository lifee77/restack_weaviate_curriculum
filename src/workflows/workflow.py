from datetime import timedelta
from pydantic import BaseModel, Field
import json
from restack_ai.workflow import workflow, import_functions, log, RetryPolicy

with import_functions():
    from src.functions.weaviate_functions import semantic_search, hybrid_search, QueryInput
    from src.functions.gemini_function_call import gemini_function_call, FunctionInputParams
    from src.functions.vector_similarity_search import vector_similarity_search
    from src.functions.text_to_braille import text_to_braille, BrailleInput
    from src.functions.text_to_audio import text_to_audio, AudioInput  # ✅ Import new function

class CurriculumInput(BaseModel):
    user_content: str = Field(default="I want to learn about coding with Python")

@workflow.defn()
class CurriculumWorkflow:
    @workflow.run
    async def run(self, input: CurriculumInput):
        try:
            function_results = []
            log.info("CurriculumWorkflow started")

            # Step 1: Use Gemini to generate curriculum ideas
            response = await workflow.step(
                gemini_function_call, 
                input=FunctionInputParams(
                    user_content=input.user_content + ". You are a helpful assistant, you have to use tools to search for books and create a curriculum for a user to learn about a topic", 
                    tools=True, 
                    structured_output=False
                ), 
                start_to_close_timeout=timedelta(seconds=120), 
                retry_policy=RetryPolicy(maximum_attempts=1), 
                task_queue="gemini"
            )

            # Step 2: Generate the final curriculum based on search results
            curriculum = await workflow.step(
                gemini_function_call, 
                input=FunctionInputParams(
                    user_content=f"Based on these results: {'; '.join(function_results)}, give me a curriculum for the user to learn about the topic. The curriculum should be a list of books that the user should read to learn about the topic.", 
                    tools=False, 
                    structured_output=True
                ), 
                start_to_close_timeout=timedelta(seconds=120), 
                retry_policy=RetryPolicy(maximum_attempts=1), 
                task_queue="gemini"
            )

            # Step 3: Generate a summary for the curriculum
            summary = await workflow.step(
                gemini_function_call, 
                input=FunctionInputParams(
                    user_content=f"Make a two-sentence summary for an audio ad of the following curriculum: {json.dumps(curriculum)}", 
                    tools=False, 
                    structured_output=False
                ), 
                start_to_close_timeout=timedelta(seconds=120), 
                retry_policy=RetryPolicy(maximum_attempts=1), 
                task_queue="gemini"
            )

            # ✅ Extract only the text content from the Gemini response
            if isinstance(summary, dict) and "candidates" in summary:
                for candidate in summary["candidates"]:
                    for part in candidate.get("content", {}).get("parts", []):
                        if isinstance(part, str):  # Ensure it's plain text
                            summary_text = part
                            break  # Stop at the first valid text result
                else:
                    summary_text = json.dumps(summary)  # Fallback: Convert dict to string
            else:
                summary_text = str(summary)  # Ensure it’s a string

            # Step 4: Convert the summary to Braille
            braille_output = await workflow.step(
                text_to_braille,
                input=BrailleInput(text=summary_text),  # ✅ Now always passing a string
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1)
            )

            # Step 5: Convert summary to Audio
            audio_output = await workflow.step(
                text_to_audio,
                input=AudioInput(text=summary_text),  # ✅ New step
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=1)
            )

            log.info(f"Braille Output: {braille_output.braille_text}")
            log.info(f"Audio Output File: {audio_output.audio_file}")

            return {
                "curriculum": curriculum["parsed"],
                "braille_summary": braille_output.braille_text,
                "audio_summary": audio_output.audio_file  # ✅ Includes audio file
            }

        except Exception as e:
            log.error(f"Error in CurriculumWorkflow: {e}")
            raise e


### ✅ NEW: Define a separate Braille workflow
class BrailleWorkflowInput(BaseModel):
    text: str = Field(default="Hello, how are you?")

@workflow.defn()
class BrailleWorkflow:
    @workflow.run
    async def run(self, input: BrailleWorkflowInput):
        try:
            log.info("BrailleWorkflow started")

            # ✅ Ensure `input.text` is a string
            text_input = str(input.text)

            # Convert text to Braille
            braille_output = await workflow.step(
                text_to_braille,
                input=BrailleInput(text=text_input),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1)
            )

            log.info(f"Braille Output: {braille_output.braille_text}")
            return {"braille_text": braille_output.braille_text}

        except Exception as e:
            log.error(f"Error in BrailleWorkflow: {e}")
            raise e
