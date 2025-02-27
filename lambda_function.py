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
        
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'},
        "body": json.dumps(return_value)
    }            