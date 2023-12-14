# NPChat
## LLM-powered NPCs for an immersive game experience

### In the world of scripted NPCs, Berkeley MIDS students joined forces to create LLM-Powered NPCs…

## Problem & Motivation
The gaming industry, worth more than $300 billion dollars, is actively improving games through the integration of AI. With companies like Unreal Engine and Unity scaling games through better graphics and innovative game design, there is a constant push towards improving player experience. Non-player characters’ (NPCs) role in games to this point has been defined as static and scripted with their role to create some small level of immersion and drive story and quests. This was not seen as a limitation for a long time. Game developers accepted the static, scripted NPC behaviors as the norm, but with recent advances in AI, there is now potential to make NPCs much more dynamic, reactive, and human-like.
 
The purpose of NPC’s is to create a world that feels realistic. If a user/gamer was able to have a realistic conversation that is customized based on how the user/gamer interacts with the game, that experience would be novel. However, current NPC’s are predictive and templated. Majority of the time, conversations with an NPC does not feel natural and pulls the user out of the “immersive experience”. Our naturally LLM-Powered NPC’s will allow for a more unique and personalized experience for the user because the experience will be tailored towards the player's interactions.
 
NPChat creates a more immersive experience for the user that will allow for more realistic and human-like dialogue, dynamic reaction and relationship building, quest generation, and more lifelike behavior. This is the next step in the evolution of NPCs in gaming. Imagine playing the same game as a friend but having a completely unique experience; this capability would open up that possibility.

## Minimal Viable Product (MVP)

The MVP provides 2 modes, interacting with an existing character or interacting with a character that you create yourself. In the first mode, the user can choose across ~100 different characters ranging from Lara Croft to Naruto Uzumaki. In the second mode, the user can input the NPCs name and biography, and proceed to interact. The second mode demonstrates that game developers can achieve the creation of NPCs with less effort.

The backend of the MVP for both modes follows the same pipeline. For both modes, the user provides an input (whether it is a question to ask the NPC or the name and biography). We employ a Retrieval Augmented Generation (RAG) pipeline to identify the answer to the question and produce a response. This input queries a database of information maintained by our team that is meant to simulate possible gaming input, which game developers could update with relevant information in the context of an actual game. The query results are then appended to the biography and user input to one of two fine-tuned language generation models (DialogStudio-T5 and LLaMa-2), where the endpoints were extracted from Amazon Sagemaker. Lastly, the model generates a response that is displayed to the user on the Streamlit app.

![Diagram of MVP](/images/MVP.png)

## Try it out!
[Click Here to try NPChat for yourself!](https://chatbot-kf5skpawx3ew3ho63mnlub.streamlit.app)

/Users/sarahhoover/Downloads/2023-12-14 13-33-15.mp4
 
## Model Evaluation
In order to generate the most appropriate response, the team initially used Rouge Scores for evaluation. Although the Rouge Scores are a good indicator of model performance, these metrics did not serve well for LLM-Powered NPC. The reason being is that the purpose of the MVP is to have a dynamic and, for the most part, unique interaction/conversation, and therefore the problem with Rouge Scores is that there is no “right” answer for conversational questions. With that being said, the team pivoted towards human annotation to evaluate DialogStudio-T5 and LLaMa-2 performance.
 
## Why 2 Models? 
Through multiple experimentation, the team noticed that the MVP was able to converse with the user using DialogStudio-T5. However, the MVP was not able to generate sensible responses. Integrating LlaMa-2 allowed the MVP to 1) converse and 2) generate logical responses to the input. 


## Meet the NPChat Team
![NPChat Rockstars](/images/team.png)

## Challenges and Future Steps 
The HuggingFace NPC Dataset:  
-Small set of main characters  
-Limited dialogue combinations  
-Combination of DialogStudio-T5 and LlaMa-2  
Ideally have one model  
Implementing the NPChat into a video game  
Models sometimes hallucinate or produce no response, even with RAG

## Acknowledgements

We would like to thank our capstone instructors, Zona Kostic and Cornelia Ilin. We are also grateful to Dr. Mark Butler and the Salesforce team for their guidance. We would also like to thank previous capstone teams [LanguageX](https://www.ischool.berkeley.edu/projects/2023/languagex) and [Tailings Identification](https://ginnyp.github.io/tailings) for providing inspiration for our website.

## Code

You can check out our code [here](https://github.com/slhoover/NPChat)

