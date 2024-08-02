import boto3
import pprint

bedrock = boto3.client("bedrock", region_name="us-east-1")
pprint.pprint(bedrock.list_foundation_models()["modelSummaries"])

pprint.pprint(bedrock.get_foundartion_model(modelIdentifier = "amazon.titan-text-express-v1"))
