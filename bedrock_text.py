import boto3
import json

bedrock = boto3.client('bedrock-runtime')

accept = 'application/json'
contentType = 'application/json'

modelId1="ai21.j2-mid-v1"
modelId2="anthropic.claude-v2"


prompt=  """prompt": "Write an email from Bob, Customer Service Manager, 
       to the customer "John Doe" 
       who provided negative feedback on the service provided by our customer support engineer"""


# リクエストBODYの指定
body_ai21_j2_mid_v1 = json.dumps({
    "prompt": prompt,
    "maxTokens": 100,
    "temperature": 0.7,
    "topP": 1,
})

body_anthropic_claude_v2 = json.dumps({
    "prompt": "Human: 富士山の高さは何メートルですか？Assistant: ",
    "max_tokens_to_sample": 200
})

response = bedrock.invoke_model(body=body_ai21_j2_mid_v1,
                                modelId=modelId1,
                                accept=accept, 
                                contentType=contentType)

# APIレスポンスからBODYを取り出す
response_body = json.loads(response.get('body').read())
# レスポンスBODYから応答テキストを取り出す
#outputText = response_body['completion']
outputText = response_body.get('completions')[0].get('data').get('text')
print(outputText)


