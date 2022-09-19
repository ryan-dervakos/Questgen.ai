from Questgen import main
import json
qgen = main.QGen()
bgen = main.BoolQGen()

def lambda_handler(event, context):
        
    body = json.loads(event['body'])
    question_type = body['question_type']

    if question_type == 'mcq':
        return_value =  { 
            'mcq' : qgen.predict_mcq(body)
        }

    if question_type == 'boolean':
        return_value =  { 
            'boolean' : bgen.predict_boolq(body)
        }

    if question_type == 'faq':
        return_value =  { 
            'boolean' : qgen.predict_shortq(body)
        }
        
    return return_value            