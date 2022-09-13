import torch
import random
from collections import OrderedDict
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from keybert import KeyBERT
import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

def MCQs_available(word,s2v):
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    
    if sense is not None:
        return True
    else:
        return False

def get_distractors(context, word):
    print('generate distractors')
    prompt = "Question:"+context+"\n\nAnswer:"+word+"\n\nList wrong choices:"
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0.8,
        presence_penalty=0
    )

    distractors = response.choices[0].text
    
    distractors_list = distractors.splitlines()

    while ("" in distractors_list):
        distractors_list.remove("")
    
    distractors_list = (list(OrderedDict.fromkeys(distractors_list)))

    return distractors_list



def edits(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def sense2vec_get_words(word,s2v):
    output = []

    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()

    word_edits = edits(word_preprocessed)

    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=15)

    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ")
        append_word = append_word.strip()
        append_word_processed = append_word.lower()
        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("","", string.punctuation))
        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word_processed)


    out = list(OrderedDict.fromkeys(output))

    return out

def get_options(answer,s2v):
    distractors =[]

    try:
        distractors = sense2vec_get_words(answer,s2v)
        if len(distractors) > 0:
            print(" Sense2vec_distractors successful for word : ", answer)
            return distractors,"sense2vec"
    except:
        print (" Sense2vec_distractors failed for word : ",answer)


    return distractors,"None"

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        word = word.strip()
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    delete_keys = []
    for k in keyword_sentences.keys():
        if len(keyword_sentences[k]) == 0:
            delete_keys.append(k)
    for del_key in delete_keys:
        del keyword_sentences[del_key]

    return keyword_sentences


def is_far(words_list,currentword,thresh,normalized_levenshtein):
    threshold = thresh
    score_list =[]
    for word in words_list:
        score_list.append(normalized_levenshtein.distance(word.lower(),currentword.lower()))
    if min(score_list)>=threshold:
        return True
    else:
        return False

def filter_phrases(phrase_keys,max,normalized_levenshtein ):
    filtered_phrases =[]
    if len(phrase_keys)>0:
        filtered_phrases.append(phrase_keys[0])
        for ph in phrase_keys[1:]:
            if is_far(filtered_phrases,ph,0.7,normalized_levenshtein ):
                filtered_phrases.append(ph)
            if len(filtered_phrases)>=max:
                break
    return filtered_phrases


def get_phrases(doc):
    phrases={}
    for np in doc.noun_chunks:
        phrase =np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase]=1
            else:
                phrases[phrase]=phrases[phrase]+1

    phrase_keys=list(phrases.keys())
    phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)
    phrase_keys=phrase_keys[:]
    return phrase_keys



def get_keywords(nlp,text,max_keywords,s2v,fdist,normalized_levenshtein,no_of_sentences):
    doc = nlp(text)
    max_keywords = 20
    kw_model = KeyBERT('all-mpnet-base-v2')

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), top_n=max_keywords, nr_candidates=10, stop_words='english')
    keywords = sorted(keywords, key=lambda x: -x[1])
    keywords = [k[0] for k in keywords]
    keywords = filter_phrases(keywords, max_keywords,normalized_levenshtein )
    phrase_keys = get_phrases(doc)

    filtered_phrases = filter_phrases(phrase_keys, max_keywords,normalized_levenshtein )

    total_phrases = keywords + filtered_phrases

    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )
    answers = []
    for answer in total_phrases_filtered:
        # if answer not in answers and MCQs_available(answer,s2v):
        if answer not in answers:    
            answers.append(answer)

    answers = answers[:max_keywords]
    return answers

def generate_questions_mcq(keyword_sent_mapping,device,tokenizer,model,sense2vec,normalized_levenshtein):
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        # text = context + " " + "answer: " + answer + " </s>"
        text = "answer: " + answer + " " + context
        batch_text.append(text)
    
    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")


    print ("Running model for generation")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,
                              attention_mask=attention_masks,
                              max_length=150)

    output_array ={}
    output_array["questions"] =[]

    for index, val in enumerate(answers):
        individual_question ={}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        Question = dec.replace("question:", "")
        Question = Question.strip()
        individual_question["question_statement"] = Question
        individual_question["question_type"] = "MCQ"
        individual_question["answer"] = val
        individual_question["id"] = index+1
        individual_question["options"] = get_distractors(Question, val)
        # individual_question["options"] =  filter_phrases(individual_question["options"], 10,normalized_levenshtein)
        index = 3
        individual_question["context"] = keyword_sent_mapping[val]

        if len(individual_question["options"])>0:
            output_array["questions"].append(individual_question)

    return output_array

def generate_normal_questions(keyword_sent_mapping,device,tokenizer,model):  #for normal one word questions
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)

    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")


    print ("Running model for generation")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,
                              attention_mask=attention_masks,
                              max_length=150)

    output_array ={}
    output_array["questions"] =[]
    
    for index, val in enumerate(answers):
        individual_quest= {}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        Question= dec.replace('question:', '')
        Question= Question.strip()

        individual_quest['Question']= Question
        individual_quest['Answer']= val
        individual_quest["id"] = index+1
        individual_quest["context"] = keyword_sent_mapping[val]
        
        output_array["questions"].append(individual_quest)
        
    return output_array

def random_choice():
    a = random.choice([0,1])
    return bool(a)
    
