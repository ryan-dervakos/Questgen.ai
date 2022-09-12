from Questgen import main
import json

def lambda_handler(event, context):
        
    body = json.loads(event['body'])
    question_type = body['question_type']
    input_text = body['input_text']

    if question_type == 'mcq':
        return_value =  { 
            'mcq' : main.QGen().predict_mcq(input_text)
        }

    if question_type == 'boolean':
        return_value =  { 
            'boolean' : main.BoolQGen().predict_boolq(input_text)
        }

    if question_type == 'faq':
        return_value =  { 
            'boolean' : main.QGen().predict_shortq(input_text)
        }
        
    return return_value            