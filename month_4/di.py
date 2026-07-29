#from pyannote.audio  import Pipeline 


import whister 
import openai
from openai import OpenAI


# the loop for our voice agent 
def voice_agent_loop():
    model = whister.load_model("base")
    client = OpenAI()







