import sys
print(sys.prefix)
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()

from autogen import ConversableAgent, UserProxyAgent

local_llm_config = {
    "config_list": [
        {
            "model": "FLlama",  # 
            "api_key": "ollama",  # 
            "base_url": "http://localhost:11434/v1",  # Your URL
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
           
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

from autogen import AssistantAgent, UserProxyAgent
import pandas as pd 
prompts_data=pd.read_csv("./CoveCs4/CoveStoryStepThree.csv")
prompts_data["FinalResponse"]=""

for i in tqdm(range(len(prompts_data))):
    
    user = ConversableAgent(
    name="User",
    system_message="",
    llm_config=local_llm_config,
    human_input_mode="NEVER",
    
)
    sysv1 = ConversableAgent(
    name="Sys",
    system_message="",
    llm_config=local_llm_config,
    human_input_mode="NEVER",
   
)

    
#   chatter=user.initiate_chats(    [
#       {
#           "recipient": sysv1,
#           "message": prompts_data.loc[i,"FinalPrompt"],
#           "max_turns": 1,
#           "summary_method": "last_msg",
#           "clear_history": False           
#        }])

  #  chat_results1 = user.initiate_chats(
  #   [
  #       {
  #           "recipient": sysv1,
  #           "message":f"""User: '{prompts_data.loc[i,"FinalPrompt"]}'\n\nAssistant: '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nAs an AI assistant, please list verification questions to check the factual accuracy of the assistant's response story above and whether it adheres to the user's instructions.""",
  #           "max_turns": 1,
  #           "summary_method":"last_msg",
  #           "clear_history": False
  #          
  #       },
  #       {
  #           "recipient": sysv1,
  #           "message":f"""Story: '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nTo verify the story above, answer each of these verification questions in the Context below.\n""",
  #           "max_turns": 1,
  #           "summary_method": "last_msg",
  #           "clear_history": False
 #           
#         }])

    chat_results1 = user.initiate_chats(
     [
         {
             "recipient": sysv1,
            # "message": f"""User: '{prompts_data.loc[i,"FinalPrompt"]}'\n\nInitial Response Story: '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nAbove is your initial response story to the user's query. Based on the verification results below, please rectify your story and provide a final, corrected story.\n\nVerification Questions: '{prompts_data.loc[i,"FinalQuestions"]}'\n\nVerification Answers: '{prompts_data.loc[i,"FinalAnswers"]}'\n\nPlease output only the final,rectified story.""",
             "message": f"""User: '{prompts_data.loc[i,"FinalPrompt"]}'\n\nAssistant (Initial Response Story): '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nVerification Questions: '{prompts_data.loc[i,"FinalQuestions"]}'\n\nVerification Answers: '{prompts_data.loc[i,"FinalAnswers"]}'\n\nBased on the verification results, please provide a final, corrected response story to the user's query.""",
             "max_turns": 1,
             "summary_method": "last_msg",
             "clear_history": False
            
         },
     ]
 )
    prompts_data.loc[i,"FinalResponse"]=chat_results1[0].chat_history[1]["content"]
   # prompts_data.loc[i,"FinalAnswers"]=chat_results1[0].chat_history[3]["content"]
    
    
prompts_data.to_csv("./CoveCs4/CoveStoryStepFour.csv")

