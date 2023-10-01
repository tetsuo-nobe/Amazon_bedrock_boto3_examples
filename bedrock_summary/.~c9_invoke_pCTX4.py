import json
import os
import sys
import boto3

from utils import print_ww

boto3_bedrock = boto3.client("bedrock-runtime")

prompt = """

Human: 下記のテキストの要約を提示して下さい。
<text>
AWS はお客様からのフィードバックをすべて取り入れ、本日 Amazon Bedrock を発表できることを嬉しく思います。\
AI21 Labs、Anthropic、Stability AI、Amazon の FM に API 経由でアクセスできるようにする新しいサービスです。 \
Bedrock は、顧客が FM を使用して生成 AI ベースのアプリケーションを構築および拡張する最も簡単な方法です。\
すべての建設業者のアクセスを民主化します。 Bedrock は、さまざまな強力な FM にアクセスする機能を提供します。
テキストと画像用 -  同じく発表している 2 つの新しい LLM で構成される Amazons Titan FM を含む \
今日、スケーラブルで信頼性が高く安全な AWS マネージド サービスを通じて。 Bedrock のサーバーレス エクスペリエンスにより、\
顧客は、やろうとしていることに適したモデルを簡単に見つけて、非公開ですぐに開始できます \
FM を独自のデータでカスタマイズし、AWS を使用してアプリケーションに簡単に統合してデプロイできます。\
インフラストラクチャ を管理する必要がなく、使い慣れたツールや機能を利用できます。\
(統合を含むさまざまなモデルをテストするための実験や、大規模な FM を管理するためのパイプラインなどの Amazon SageMaker ML 機能を使用します)。
</text>

Assistant:"""

body = json.dumps({"prompt": prompt,
                 "max_tokens_to_sample":4096,
                 "temperature":0.5,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 
                  

modelId = 'anthropic.claude-v2' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())

print_ww(response_body.get('completion'))


