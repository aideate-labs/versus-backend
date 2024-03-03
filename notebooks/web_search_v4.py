# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 0: Setup and installation

# + active=""
# !pip3 install python-dotenv
# !pip3 install azure
# !pip3 install azure-cognitiveservices-search-websearch
# !pip3 install keyboard
# !pip3 install selenium
# !pip3 install webdriver_manager

# +
from openai import OpenAI
import pprint

OPENAI_ORG_KEY = 'org-CJBqTlt7yBaKH17EzphQivGs'
#OPENAI_API_KEY = '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'
OPENAI_API_KEY = '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'
client = OpenAI(api_key = OPENAI_API_KEY , organization = OPENAI_ORG_KEY)
#model="gpt-4-1106-preview",

# + active=""
# completion = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {"role": "system", "content": "You are a personal assistant and a product expert. You wont be provided any documents for retrieval. Your job is to fetch the latest data from the internet and present the comparisons of the product in a tabular format."},
#     {"role": "user", "content": "Compare dishwashers 2023"}
#     ]
# )
#
# print(completion.choices[0].message)

# +
import os
import asyncio
import requests
import time
import json
import openai
import keyboard

from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv
from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 1: Content extraction

# +
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import re
from html import unescape
import nltk
from nltk.tokenize import sent_tokenize

def get_webpage_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        return f"Error during requests to {url} : {str(e)}"


def extract_main_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extended list of CSS selectors targeting common main content areas
    selectors = [
        'article', 'main', 
        'div.content', 'div#content', 
        'div.post', 'div.article', 
        'div.main-content', 'section.content',
        'p'
    ]
    
    best_candidate = None
    best_length = 0
    
    for selector in selectors:
        for element in soup.select(selector):
            text = element.get_text()
            text_length = len(text)
            if text_length > best_length:
                best_candidate = text
                best_length = text_length

    # If no significant content is found, fallback to broader search
    if not best_candidate or best_length < 100:  # Arbitrary minimum length
        for element in soup.find_all(['div', 'section','p'], limit=10):  # Limit search to first few divs/sections
            text = element.get_text()
            text_length = len(text)
            if text_length > best_length and text_length > 200:  # Adjust criteria as needed
                best_candidate = text
                best_length = text_length

    return best_candidate if best_candidate else "Main content not found."

def get_dynamic_webpage_content(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    #service = webdriver.Service(ChromeDriverManager().install())
    #service = Service(ChromeDriverManager().install())

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        rendered_html = driver.page_source
        driver.quit()
        return rendered_html
    except Exception as e:
        return f"Error loading dynamic content: {str(e)}"


def preprocess_text(text):
    # Decode HTML entities
    text = unescape(text)

    # Remove unwanted characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets (often metadata or links)
    text = text.strip()  # Remove leading/trailing whitespace

    return text



def segment_text(text):
    try:
        nltk.download('punkt', quiet=True)  # Ensure the tokenizer is available
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        return f"Error in text segmentation: {str(e)}"


def is_dynamic(url):
    """
    A heuristic approach to guess if a webpage is dynamic.
    This function checks for the presence of script tags which could indicate dynamic content.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        scripts = soup.find_all('script')
        # Heuristically determine if a page might be dynamic by the number of script tags
        if len(scripts) > 5:  # Arbitrary threshold
            return True
        else:
            return False
    except Exception as e:
        #print(f"Error checking if page is dynamic: {str(e)}")
        return False  # Default to false in case of error


def extract_webpage_content(url):
    html_content = ''
    if is_dynamic(url):
        #print("Content is dynamic...........\n")
        html_content = get_dynamic_webpage_content(url)
        #print("html content len: ",len(html_content))
    else:
        #print("Content is not dynamic...........\n")
        html_content = get_webpage_content(url)
        #print("html content len: ",len(html_content))

    #if 'Error' in html_content:
       # return html_content  # Return the error message if an error occurred

    main_content = extract_main_content(html_content)
    #print("main content len: ", len(main_content))
    if main_content and not 'Error' in main_content:
        cleaned_text = preprocess_text(main_content)
        segmented_text = segment_text(cleaned_text)
        #print("segmented text len: ", len(segmented_text))
        return str(segmented_text)
    else:
        return str(main_content) or "Error: Main content not found."

# Example Usage
#url = 'https://www.mieleusa.com/e/dishwashers-1015063-c'
#extracted_content = extract_webpage_content(url)
#print(extracted_content)








# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 2: README

# +
# NOTE: Bing search is free for 1000 queries per month!
# NOTE: OpenAI is *not cheap* particularly using assisistants - just keep in mind!
#       GPT-4 Turbo (gpt-4-1106-preview) is set here, but you can use GPT-3.5 Turbo (gpt-3.5-turbo-1106) which is 10x cheaper

# Implementation Overview:
# - Creates an OpenAI assistant instance set up to call Bing-search and analysis functions (funcs defined in this script)
# - Uses Bing Search API to conduct internet searches based on the processed user query. NOTE: Can easily be swapped to use the Bing Custom Search service (aka resource) instead (which allows you to define what sites it can search, and what sites it can't search)

# Bing API & Service (resource) Setup:
# 1. Sign Up for Azure
#   - If you don't already have one, create a Microsoft Azure account at Azure Portal.
# 2. Go to the Azure Marketplace within the Azure Portal
# 3. Create a new Bing Search resource 
#   - If you wish to limit the search window to certain URLs or domains, then see comments listing 'Bing Custom Search' below.
# 4. Copy the API key from the Azure Portal

# Overview of the Process:
# 1: User Request: 
#   - The user provides a plain-english request for information (e.g. "What are the best stonks to buy right now?").
# 2. Automated Bing Web-Search: 
#   - OpenAI assistant generates a query to search Bing
#   - Implements the Bing Web Search API to conduct internet searches based on the processed user query.
#   - If the user request isn't clear enough to form a Bing-search query from (example query: "sup man?"), then the assistant will (likely) respond seeking more information.
# 3. Automated Search Result Processing: 
#   - Analyzes and processes the search results obtained from Bing, again using the OpenAI assistant instance.
#   - The assistant will then provide a summary of the search results, tailored to answer the user's initial query.
# Result Analysis and Response: Provides a summary or analysis of the search results, tailored to answer the user's initial query.

# NOTE (references and notes)
#  
# Bing web search (Azure) API (Use Azure marketplace and search for "Bing Search" or "Bing Custom Search" - avoid the old Bing API)
#   CURRENT AZURE API REF: https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/quickstarts/python
#   DEPRECATED API REF: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/sdk/web-search-client-library-python
# 
# Tavily web-search for LLMs (alternative option - probably not as good as Bing search)
#   https://docs.tavily.com/docs/tavily-api/python-sdk
#
# OpenAI Tools (assistants):
#   https://platform.openai.com/docs/assistants/tools
#   https://platform.openai.com/docs/assistants/tools/function-calling
#   RECENT: https://medium.com/@nilakashdas/how-to-build-smart-assistants-using-open-ai-apis-ebf9edc42084 
#   https://medium.com/@assafelovic/how-to-build-an-openai-assistant-with-internet-browsing-ee5ad7625661
#   NOTE, share results with community:
#   https://community.openai.com/t/new-assistants-browse-with-bing-ability/479383/12
#   Multi-function assistants:
#   https://dev.to/esponges/build-the-new-openai-assistant-with-function-calling-52f5
# 
# Use a connection to Azure OpenAI on your data.
#   https://learn.microsoft.com/en-us/microsoft-copilot-studio/nlu-generative-answers-azure-openai

# TODO: 
#   Have LLM scrape data from sites linked by the Bing web search, and then analyze the *scraped data* (rather than raw search results) to answer the user's question.
#   Consider using local LLM for generating Bing search query (it creates an optimized keyword search query from user's plain english request) 
#   Consider using local LLM for performing Bing search analysis (for answering user's question)





# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 3: Configurations

# +
############################################################################################################
### CONFIGURATIONS 
############################################################################################################

# Load environment variables
load_dotenv()

# OpenAI API Key
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) #commented
client = OpenAI(api_key = OPENAI_API_KEY , organization = OPENAI_ORG_KEY)

# NOTE: OLD BING API fields
# subscription_key = "79f43664b4a343e38b4017f758ee80f3"
# search_client = WebSearchClient(endpoint="https://api.bing.microsoft.com/", credentials=CognitiveServicesCredentials(subscription_key))

# NOTE: NEW BING API fields (API migrated to azure marketplace)
# custom_config_id = "define this if you are using 'Bing Custom Search' service (aka resource) instead of 'Bing Search'"
searchTerm = "microsoft"
# NOTE: This URL is not the same as the one listed in the Azure resource portal. It has the additional v7.0/search? to specify the resource function.
url = 'https://api.bing.microsoft.com/v7.0/search?' #  + 'q=' + searchTerm + '&' + 'customconfig=' + custom_config_id
#url = "https://api.bing.microsoft.com/"

# OpenAI Model Configuration
base_model = "gpt-4-1106-preview"
max_tokens = 15000 
temperature = 0.2

u_request = ""
s_query = ""
s_results = ""
run = None


# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 4: OpenAI functions
# -

# #### 4.1: Create run_bing_search function that will be called in the next perform_bing_search() function

# +
############################################################################################################
### ANALYSIS: Perform a Bing search and process the results
############################################################################################################

def run_bing_search(search_query):
  # Returns data of type SearchResponse 
  # https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-search-websearch/azure.cognitiveservices.search.websearch.models.searchresponse?view=azure-python
  # Bing search set-up on - Azure portal - https://portal.azure.com/#home
    try:
        subscription_key = "67d46838a54b4c2eb296ee67f0d8b517"
        base_url = "https://api.bing.microsoft.com/v7.0/search?"
        #base_url = "https://api.bing.microsoft.com/"
        encoded_query = quote_plus(search_query)
        bing_search_query = base_url + 'q=' + encoded_query # + '&' + 'customconfig=' + custom_config_id --> uncomment this if you are using 'Bing Custom Search'
        r = requests.get(bing_search_query, headers={'Ocp-Apim-Subscription-Key': subscription_key})
    except Exception as err:
        print("\n####  Encountered exception. {}".format(err))
        raise err
  
  # Old API
  #try:
  #  web_data = search_client.web.search(query=search_query)
  #except Exception as err:
  #  print("Encountered exception. {}".format(err))
  #  raise err

    response_data = json.loads(r.text)
    results_text = ""
    for result in response_data.get("webPages", {}).get("value", []):
        results_text += result["name"] + "\n"
        results_text += result["url"] + "\n"
        results_text += result["snippet"] + "\n"
        #results_text += result["text"] + "\n\n"
        #print(f"Title: {result['name']}")
        #print(f"URL: {result['url']}")
        #print(f"Snippet: {result['snippet']}\n")
        #print(f"Title: {result}")
    

        extracted_content = extract_webpage_content(result["url"])
        results_text += extracted_content + "\n\n"


    
    #print("\n###### Results_text: #########\n", results_text)
    return results_text #---- original

    
# -

# #### 4.2: OPENAI FUNCTIONS: Functions to perform a Bing search and process the results

# +
############################################################################################################
### OPENAI FUNCTIONS: Functions to perform a Bing search and process the results
############################################################################################################

# OPENAI FUNCTION: Function to perform a Bing search
def perform_bing_search(user_request):
  global u_request
  global s_query
  global s_results

  u_request = user_request
  print(f"\n### Generating a search_query for bing based on this user request: {user_request}")
  openai_prompt = "Generate a search-engine query to satisfy this user's request: " + user_request
  response = client.chat.completions.create(
      model=base_model,
      messages=[{"role": "user", "content": openai_prompt}],
  )
  # Get the response from OpenAI
  bing_query = response.model_dump_json(indent=2)
  s_query = bing_query
  print(f"\n### Bing search query: {bing_query}. \n### Now executing the search...")

    # Calling function run_bing_search()
  bing_response = run_bing_search(user_request)
  s_results = bing_response
  return bing_response
    

# OPENAI FUNCTION: Function to process Bing search results
def process_search_results(search_results):
  global u_request
  global s_query
  global s_results

  print(f"\n#### Analyzing/processing Bing search results")

  # Use GPT to analyze the Bing search results
  #prompt = f"Analyze these Bing search results: '{s_results}'\nbased on this user request and append insights to a result table that has product names from the search result as column headers,  comparison factors as row headers, and values as insights for those products x factors combination. Do not skip any information like pricing and availability for any reason.   : {u_request}"
  prompt = f"Analyze these Bing search results: '{s_results}'\nbased on this user request and create a knowledge base of all data possible of the product in the given geography. Do not skip any information like pricing and availability for any reason. : {u_request}"

    
  response = client.chat.completions.create(
      model=base_model,
      messages=[{"role": "user", "content": prompt}],
  )
  analysis =  response.choices[0].message.content.strip()

  print(f"\n############################ ANALYSIS ############################ - process_search_results: {analysis}")
  # Return the analysis
  return analysis





# -

# ## Step 5: OpenAI assitant run management

# +
############################################################################################################
### OPENAI ASSISTANT RUN MANAGEMENT
############################################################################################################

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call, run, tool_output_array=None, func_override=None):
    global s_results
    print(f"\n#### Submitting tool outputs for thread_id: {thread_id}, run_id: {run_id}, tools_to_call: {tools_to_call}")
    if tool_output_array == None:
      tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = func_override if func_override else tool.function.name
        function_args = tool.function.arguments

        if function_name == "perform_bing_search":
            print("\n####[FUNCTION CALL #1] perform_bing_search()...")
            output = perform_bing_search(user_request = json.loads(function_args)["user_request"])

        elif function_name == "process_search_results":
            print("\n####[FUNCTION CALL #2] process_search_results()...")
            output = process_search_results(json.loads(function_args)["search_results"]) #search_results = s_results) #json.loads(function_args)["search_results"]) #(search_results = s_results) 

        if output:
          print("\n####[FUNCTION RESULT] Appending tool output array...")
          tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )


# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    message = ""
    print("\n=========== ASSISTANT RESPONSE ==========\n")
    for msg in messages:
      if msg.role == "assistant":
        print(f"{msg.role}: {msg.content[0].text.value}")
        message += f"{msg.role}: {msg.content[0].text.value}\n"
    
    return message

# Initialize the assistant and its features and tools
assistant = client.beta.assistants.create(
  instructions="""You are a Q&A bot who performs web searches to respond to user queries. 
  Call function 'perform_bing_search' when provided a user query.
  Call function 'process_search_results'.
  Call function 'perform_bing_search' again if the search results do not contain the info needed to answer the user query.
  """,
    # Call function 'process_search_results' if the search results contain the info needed to answer the user query.
  model=base_model,
  tools=[
    {
      "type": "code_interpreter"

    },
    {
    "type": "function",
    "function": {
      "name": "perform_bing_search", # Function itself should run a GPT OpenAI-query that asks the OpenAI to generate (and return) a Bing-search-query.
      "description": "Determine a Bing search query from the user_request for specified information and execute the search",
      "parameters": {
        "type": "object",
        "properties": {
          "user_request": {"type": "string", "description": "The user's request, used to formulate a Bing search message"},
        },
        "required": ["user_request"]
      }
    }
  }, 
  {
    "type": "function",
    "function": {
        "name": "process_search_results", # Function itself should send the Bing search results to openai to assess the results, and then return the results of that assessment to the user.
        #"description": "Compare top products in the US and provide recommendations",
        #"description": "Analyze Bing search results and return a summary of the results that most effectively answer the user's request",
        "description": "Analyze Bing search results and return the result",
        #"description": "Analyze Bing search results and return a summary of the results in a tabular form with products as columns and comparison factors as rows",
        "parameters": {
            "type": "object",
            "properties": {
                "search_results": {"type": "string", "description": "The results from the Bing search to analyze"},
            },
            "required": ["search_results"]
        }
    } 
  }
]
)
assistant_id = assistant.id
print(f"Assistant ID: {assistant_id}")


# Create a thread
thread = client.beta.threads.create()
print(f"Thread: {thread}")

# -

# ## Step 6: Assistant run

# + active=""
# %%time
#
# # Ongoing conversation loop
#
# while True:
#       
#     user_prompt = input("\nYour request: ")
#     prompt_additional = "Structure the insights in a result table that has product names from the search result as column headers,  comparison factors as row headers, and values as insights for those products x factors combination." 
#
#     prompt = user_prompt + prompt_additional
#     if prompt.lower() == 'e':
#         print("Exiting conversation")
#         break
#
#     status = "na"
#     
#     #while status != "completed":
#       # Create a message and run
#     message = client.beta.threads.messages.create(
#         thread_id=thread.id,
#         role="user",
#         content=prompt,
#     )
#     run = client.beta.threads.runs.create(
#         thread_id=thread.id,
#         assistant_id=assistant_id,   
#     )
#     print(f"Run ID: {run.id}")
#     # Wait for run to complete
#     
#     run = wait_for_run_completion(thread.id, run.id)
#     while run.status == 'requires_action':
#         print("Run requires action 1")
#         run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls, run) # **error on this line**
#         run = wait_for_run_completion(thread.id, run.id)
#
#         time.sleep(1)
#     if run.status == 'failed':
#         print(run.error)
#         continue
#     # Print messages from the thread
#     #prompt = print_messages_from_thread(thread.id)
#     print_messages_from_thread(thread.id)
#     time.sleep(1)

# + active=""
# import streamlit as st
# from langchain.llms import OpenAI
#
# st.title('Compare anything')
#
# openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
#
# def generate_response(input_text):
#     llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#     st.info(llm(input_text))
#
# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     if not openai_api_key.startswith('sk-'):
#         st.warning('Please enter your OpenAI API key!', icon='⚠')
#     if submitted and openai_api_key.startswith('sk-'):
#         generate_response(text)
# -

# ## Step 6.B. Streamlit run

# +
import streamlit as st
from langchain.llms import OpenAI

st.title('Compare anything')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Ongoing conversation loop

user_prompt = st.text_area('Enter text:', '', key='text_input_1')

#while True:
if user_prompt:
      
    #user_prompt = input("\nYour request: ")
    #user_prompt = st.text_area('Enter text:', '', key='text_input_1')
    
    prompt_additional = "Structure the insights in a result table that has product names from the search result as column headers,  comparison factors as row headers, and values as insights for those products x factors combination." 

    prompt = user_prompt + prompt_additional
    
    #if prompt.lower() == 'e':
     #   print("Exiting conversation")
      #  break

    status = "na"
    
    #while status != "completed":
      # Create a message and run
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,   
    )
    print(f"Run ID: {run.id}")
    # Wait for run to complete
    
    run = wait_for_run_completion(thread.id, run.id)
    while run.status == 'requires_action':
        print("Run requires action 1")
        run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls, run) # **error on this line**
        run = wait_for_run_completion(thread.id, run.id)

        time.sleep(1)
    if run.status == 'failed':
        print(run.error)
        #continue
    # Print messages from the thread
    #prompt = print_messages_from_thread(thread.id)
    
    message = print_messages_from_thread(thread.id)
    st.markdown(message)
    time.sleep(1)
# -

# ## END

# + active=""
# Compare dishwashers in USA in 2023. For the comparison create a knowledge base of all data possible of the product in the given geography. Structure the insights in a result table that has product names from the search result as column headers,  comparison factors as row headers, and values as insights for those products x factors combination. Do not skip any information like pricing and availability for any reason. Fill all the details to the best of your ability. 
#
#

# + active=""
# ############################ ANALYSIS ############################ - process_search_results: Based on the user request, here is a comprehensive knowledge base of the best dishwashers in the US for 2023, including information on pricing, availability, features, and expert evaluations from various sources:
#
# 1. **Bosch SHP78CM5N** (Bosch 800 Series):
#     - Excellent cleaner with versatile racks
#     - Fast cycle times
#     - Features PrecisionWash with PowerControl, stainless steel tub, EasyGlide racks, CrystalDry with zeolite
#     - Noted for being a little expensive but providing top-notch performance
#     - Pricing: Approximately $1,299.99 at major retailers like Best Buy.
#     - Rated 4.7 on one of the reviewed lists.
#
# 2. **Miele G 5266 SCVi SFP**:
#     - Known for best drying performance
#     - Stainless steel finish, third rack, AutoOpen drying, ExtraClean & ExtraDry features
#     - Rated 4.7, available at retailers like Abt for around $1,749.00
#     - Identified for providing impressive cleaning power with limited significant downsides.
#
# 3. **Samsung DW80R9950UT**:
#     - Low decibel level, approximately 39 dBA, making it very quiet
#     - Short cycles, a third rack, and a fingerprint-resistant stainless steel finish
#     - Pricing varies but can be found for approximately $1,299.00 (subject to change)
#
# 4. **Beko DUT25401X**:
#     - Best value dishwasher offering excellent performance at a low price point
#     - Features include an adjustable upper rack and energy-efficient performance
#     - Difficult control panel but rated 4.5 for value
#     - Can be purchased from Appliances Connection
#
# 5. **Bosch SHEM63W55N** (Bosch 300 Series):
#     - Best overall dishwasher according to several reviews
#     - Holds 16 place settings and features a quieter operation at 44 dBA
#     - Comes with 3 racks for customizing loads
#     - Pricing: Approximate retail price of $800 at Best Buy
#
# 6. **KitchenAid KDFE204KPS**:
#     - Best for quiet operation at 47 decibels
#     - Features auto-adjust for soil level
#     - Affordable and efficient, retail price around $700 at Best Buy
#
# 7. **LG LDT7808BD**:
#     - Best dishwasher for smart features and modern design
#     - Wi-Fi-enabled for mobile control and LED lights inside
#     - Pricing is on the higher end at approximately $1,299 at Best Buy
#
# 8. **Samsung DW50T6060US**:
#     - Best compact dishwasher option
#     - Quiet with auto-release door
#     - Less capacity due to compact design
#     - Available at retailers like Walmart for around $899 to $1,098
#
# 9. **Whirlpool WDF520PADM**:
#     - Best budget dishwasher under $500
#     - Features in-door silverware rack and various wash cycles, including sanitize and heated dry
#     - Available in stainless steel finish
#     - Pricing around $450 to $599 at Walmart and Best Buy
#
# Remember, prices and availability are subject to change and can vary by retailer and region. It's recommended to check multiple retailers and look out for sales and discounts that can offer significant savings. Additionally, most of these dishwashers come with energy efficiency certifications like Energy Star, adding long-term value in terms of reduced utility costs.
#
# ####[FUNCTION RESULT] Appending tool output array...
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: completed
#
# =========== ASSISTANT RESPONSE ==========
#
# assistant: The search results from Forbes Vetted provide an overview of some of the best dishwashers for 2023, along with their key features and comparison factors. Let's structure this information into a results table:
#
# ### Comparison Table of Best Dishwashers in the US 2023
#
# | Comparison Factors    | Bosch 100 Series       | GE Built-In Tall Tub    | Bosch 500 Series     | KitchenAid PrintShield | Samsung Smart Linear Wash | Miele G7316SCU      | LG QuadWash          |
# |-----------------------|----------------|-----------------|-----------------|---------------------|-------------------------|--------------------|---------------------|
# | Position              | Best Overall   | Best Budget     | Best Smart      | Best 3rd Rack       | Quietest                | Best High-End      | Most Efficient      |
# | Price (Approx.)       | $579-$711      | $400            | $1,099          | $900-$1,215         | $800-$1,199             | $2,149-$2,687      | $799-$999           |
# | Dimensions (inches)   | 33.9 x 23.6 x 23.8 | 34.6 x 23.8 x 24 | 23.7 x 23.5 x 33.8 | 23.8 x 34 x 24         | 25 x 33.9 x 23.9      | 23.6 x 22.4 x 33.7 | 24.6 x 23.8 x 34.5 |
# | Noise Level (dBA)     | 50             | 52              | 44              | 44                   | 39                      | 45                 | 46                 |
# | Capacity              | 14 place settings | 14 place settings | 16 place settings | 16 place settings     | 15 place settings      | 16 place settings  | 15 place settings  |
# | Finishes Available    | 3              | 4               | 3               | 2                    | 4                       | 1                  | 2                   |
# | Racks                 | 3              | 2 (+ optional 3rd) | 3               | 3                    | 3                       | 3                  | 3                   |
# | Energy Star Certified | No             | Yes             | Yes             | No                   | Yes                     | Yes                | Yes                 |
# | Unique Features       | Sanitize cycle, PureDry technology | Steam, sanitize, Dry Boost cycles | AutoAir Dry, app features | Advanced 3rd rack design, ProWash cycle | AquaBlast jets, auto door opening | AutoDos, luxury design | QuadWash technology |
#
# This table captures a snapshot of the top dishwashers as recommended by Forbes Vetted, highlighting product names, comparison factors, and their corresponding insights. Keep in mind that availability, prices, and features can vary based on location and updates from manufacturers. Always check with retailers for the most current information before making a purchase.
#
# -





# + active=""
# ############################ ANALYSIS ############################ - process_search_results: After analyzing the provided Bing search results, I've created a comparison table for the best dishwashers in the USA for 2023 based on the insights gathered. Note that some specific details, like pricing and availability, were not always provided in the search results, so they are only included for the products that had this information available.
#
# | Comparison Factors       | Bosch 300 Series SHEM63W55N                         | Whirlpool WDF520PADM                                   | LG LDT7808BD                           | Samsung DW50T6060US              | Frigidaire FGID2476SF                     |
# |--------------------------|-----------------------------------------------------|--------------------------------------------------------|----------------------------------------|--------------------------------|-------------------------------------|
# | Estimated Price          | $945 - $1,049                                       | $460 - $567                                            | $899 - $1,299                          | $810 - $1,098                   | $450 - $549                           |
# | Sound Level (dBA)        | 44 dBA                                              | 55 dBA                                                 | 42 dBA                                 | 46 dBA                         | 49 dBA                               |
# | Place Settings           | 16                                                  | 14                                                     | Not specified                          | 8 (compact design)             | Up to 14                             |
# | Energy Efficiency        | Energy Star-certified                               | Energy Star-certified                                  | Energy Star-certified                  | Energy Star-certified          | Energy Star-certified                 |
# | Rack Adjustability       | 3 racks, Rackmatic adjustable upper rack, FlexSpace | 2 racks, in-door silverware rack                      | 3 racks, adjustable upper rack         | 2 racks, adjustable top rack  | Not specifically mentioned            |
# | Smart Features           | Not mentioned                                       | Not mentioned                                          | Wi-Fi-enabled, LED light, smart control| Not mentioned                  | Not mentioned                         |
# | Cycles / Special Features| 5 wash cycles including Speed60                     | Varies, "1-hour Wash" mentioned                       | Various, with steam cycle mentioned    | Auto-release door, 5 cycles   | BladeSpray Arm, EvenDry, sanitize cycle|
# | Design / Finish          | Stainless steel                                     | Stainless steel, available in other finishes          | Stainless steel                        | Stainless steel               | Stainless steel, smudge-proof finish |
# | Reviews / Ratings        | 4.6/5 (Home Depot, 5,053 reviews), 4.3/5 (Lowe's)   | 4.3/5 (Home Depot, 13,461 reviews), 4.3/5 (Lowe's)    | 4.3/5 (Home Depot), 4.3 (Lowe's)       | 4.7/5 (Home Depot, 60 reviews)| 4.3/5 (Home Depot), 4.3/5 (Lowe's)    |
# | Availability             | Available at major retailers                        | Available at major retailers                           | Available at major retailers           | Available at major retailers | Available at major retailers          |
#
# Please bear in mind that this table uses the most current information available from the search results, which as of my last update, is accurate for early 2023. Prices and availability are subject to change, so it's always best to check with retailers for the most up-to-date information before making a purchase.
#
# ####[FUNCTION RESULT] Appending tool output array...
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: in_progress
# Current run status: completed
#
# =========== ASSISTANT RESPONSE ==========
#
# assistant: After analyzing the provided Bing search results, the comparison table for the best dishwashers in the USA for 2023 based on the insights gathered is as follows:
#
# | Comparison Factors        | Bosch 100 Series 24-Inch Front Control                                                                | GE 24-Inch Top Control Dishwasher                                |
# |---------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
# | Product Name              | Bosch 100 Series                                                                                     | GE 24-Inch Top Control Dishwasher                                 |
# | Price                     | $579 at Lowe's, $650 at The Home Depot, $711 at Wayfair                                               | Under $400                                                        |
# | Dimensions (inches)       | 33.9 x 23.6 x 23.8                                                                                   | 34.6 x 23.8 x 24                                                  |
# | Sound Level (dBA)         | 50                                                                                                   | 52                                                                |
# | Capacity (Place Settings) | 14                                                                                                   | 14                                                                |
# | Finishes Available        | 3                                                                                                    | 4                                                                 |
# | Racks                     | 3                                                                                                    | 2                                                                 |
# | Energy Star Certified     | No                                                                                                   | Yes                                                               |
# | Key Features              | Sanitizing option, PureDry technology, half-load cycle option, fingerprint-resistant finish          | Steam, sanitize, Dry Boost cycles, delay start option, optional third rack |
# | Pros                      | Three racks, sleek design, quiet operation, convenient sanitize cycle                                 | Affordable, spacious, optional third rack, effective drying       |
# | Cons                      | Bottom rack may slide off track                                                                       | May not perform as well on dishes with dried-on food, third rack is an additional purchase |
# | Availability              | Available at Lowe's, The Home Depot, Wayfair                                                          | Available at Best Buy, Lowe's, The Home Depot                     |
#
# This table structures the comparison of the first two dishwasher models from the Forbes Vetted list of best dishwashers for 2023 with the necessary headers for adding additional models as the data becomes available. Pricing and availability are based on the specific details found in the search results. The table is ready to be expanded with other dishwasher models after processing the remaining content.
#
# Your request:  e
# Exiting conversation
# CPU times: user 4.27 s, sys: 1.52 s, total: 5.79 s
# Wall time: 10min 49s
#
# Selection deleted
#
# -


