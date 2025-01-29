from datetime import timedelta
from pydantic import BaseModel, Field
import json
from restack_ai.workflow import workflow, import_functions, log, RetryPolicy

with import_functions():
    from src.functions.weaviate_functions import semantic_search, hybrid_search, QueryInput
    from src.functions.gemini_function_call import gemini_function_call, FunctionInputParams
    from src.functions.vector_similarity_search import vector_similarity_search
    from src.functions.text_to_braille import text_to_braille, BrailleInput  # ✅ New import

class CurriculumInput(BaseModel):
    user_content: str = Field(default='I want to learn how about coding with Python')

@workflow.defn()
class CurriculumWorkflow:
    @workflow.run
    async def run(self, input: CurriculumInput):
        try:
            function_results = []
            log.info("CurriculumWorkflow started")
            
            ## Agentic RAG: We use the gemini_function_call function to search for books and create a curriculum for a user to learn about a topic
            
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
            
            # Check if any function calls were made
            if not response["candidates"] or all(part["functionCall"] is None for candidate in response["candidates"] for part in candidate["content"]["parts"]):
                log.error("No function calls were made")
            else:
                # Process function calls from response
                for candidate in response["candidates"]:
                    for part in candidate["content"]["parts"]:
                        if "functionCall" in part and part["functionCall"]:
                            func_name = part["functionCall"]["name"]
                            func_args = part["functionCall"]["args"]["user_content"]

                            if func_name == "hybrid_search":
                                hybrid_result = await workflow.step(hybrid_search, input=QueryInput(user_content=func_args))
                                function_results.append(f"hybrid_search result: {str(hybrid_result)}")
                            elif func_name == "semantic_search": 
                                semantic_result = await workflow.step(semantic_search, input=QueryInput(user_content=func_args))
                                function_results.append(f"semantic_search result: {str(semantic_result)}")
                            elif func_name == "vector_similarity_search":
                                vector_result = await workflow.step(vector_similarity_search, input=QueryInput(user_content=func_args))
                                function_results.append(f"vector_similarity_search result: {str(vector_result)}")
            
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
            
            ## Workshop Step 7: Add a workflow step to get a summary of the curriculum
            
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

            ## ✅ NEW STEP: Convert the Summary into Braille
            braille_output = await workflow.step(
                text_to_braille,
                input=BrailleInput(text=summary),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1)
            )

            log.info(f"Braille Output: {braille_output.braille_text}")

            return {
                "curriculum": curriculum["parsed"],
                "braille_summary": braille_output.braille_text  # ✅ Now includes Braille summary
            }

        except Exception as e:
            log.error(f"Error in CurriculumWorkflow: {e}")
            raise e
