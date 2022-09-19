from Questgen import main
import json

def lambda_handler(event, context):
        
    body = json.loads(event['body'])
    question_type = body['question_type']

    if question_type == 'mcq':
        return_value =  { 
            'mcq' : main.QGen().predict_mcq(body)
        }

    if question_type == 'boolean':
        return_value =  { 
            'boolean' : main.BoolQGen().predict_boolq(body)
        }

    if question_type == 'faq':
        return_value =  { 
            'boolean' : main.QGen().predict_shortq(body)
        }
        
    return return_value            