import pandas as pd
import unicodedata
import re

def remove_duplicated_wiki(wiki):
    """
    wikipedia_documents.json에서 "text" 항목의 중복 데이터를 제거합니다.

    Args:
        wiki (dict): dict 형태의 wikipedia_documents를 입력받습니다.

    Returns:
        dict: 중복 데이터를 제거한 dict 형태의 wikipedia_documents를 반환합니다.
    """
    wiki_df = pd.DataFrame(wiki).transpose()
    remove_df = wiki_df.drop_duplicates(["text"], keep="first")
    
    return remove_df.to_dict("index")


def remove_wiki_less_than_percents(wiki, percents=50):
    """
    wikipedia_documents.json의 "text"에서 한국어 비율이 사용자 설정 비율보다 적은 데이터를 삭제합니다.
    이는 retriever가 올바르지 않은 문서를 reader에게 제공하는 일이 없도록 하기 위해 사용됩니다.

    Args:
        wiki (dict): dict 형태의 wikipedia_documents를 입력받습니다.
        percents (int, optional): 제거할 비율을 설정합니다. Defaults to 50.

    Returns:
        dict: 한국어 비율이 특정 비율 이하인 text 데이터를 제거한 dict 형태의 wikipedia_documents를 반환합니다.
    """
    wiki_df = pd.DataFrame(wiki).transpose()
    wiki_df["kor_ratio"] = "None"

    for idx, text in enumerate(wiki_df["text"]):
        processed_text = re.sub(r"[\n\s]", "", text)
        p = re.compile("[가-힣]")
        wiki_df.iloc[idx, 8] = len(p.findall(processed_text)) / len(processed_text) * 100
    
    remove_index = wiki_df[wiki_df["kor_ratio"] < percents].index
    remove_df = wiki_df.drop(remove_index).drop(labels="kor_ratio", axis=1)

    return remove_df.to_dict("index")


def normalize_data_context(dataset):
    """
    dataset의 "context" 데이터를 반각 문자로 변환합니다.

    Args:
        dataset (DatasetDict): DatasetDict 형태의 train or valid data를 입력받습니다.
    """
    def normalize(data):
        return {"context": unicodedata.normalize("NFKC", data["context"])}
    
    return dataset.map(normalize)


def normalize_data_question(dataset):
    """
    dataset의 "question" 데이터를 반각 문자로 변환합니다.

    Args:
        dataset (DatasetDict): DatasetDict 형태의 train or valid data를 입력받습니다.
    """
    def normalize(data):
        return {"question": unicodedata.normalize("NFKC", data["question"])}
    
    return dataset.map(normalize)


def normalize_data_answer(dataset):
    """
    dataset의 "answers"의 "text" 데이터를 반각 문자로 변환합니다.

    Args:
        dataset (DatasetDict): DatasetDict 형태의 train or valid data를 입력받습니다.
    """
    def normalize(data):
        data["answers"]["text"][0] = unicodedata.normalize("NFKC", data["answers"]["text"][0])
        
        return {"answers": data["answers"]}
    
    return dataset.map(normalize)


def normalize_wiki(wiki):
    """
    wikipedia_documents의 "text" 데이터에 대해 반각 문자로 변환합니다.

    Args:
        wiki (list): list형태의 wikipedia_documents의 "text" 내용을 입력받습니다.

    Returns:
        list: 반각 문자로 변환된 list를 반환합니다. 
    """
    for idx in range(len(wiki)):
        wiki[idx] = unicodedata.normalize("NFKC", wiki[idx])
    
    return wiki


def remove_special_char(text):
    cp = re.compile('\\\\n|\*|\\n|\\|#|\"')
        
    text = re.sub(r"[“”‘’]", "\'", text)
    # text = re.sub(r"[〈<＜「≪《『]", "<", text)
    # text = re.sub(r"[〉>＞」≫》』]", ">", text)
    text = cp.sub('', text)
    
    return text


def remove_special_char_wiki(wiki):
    for i in range(len(wiki)):
        wiki[i] = remove_special_char(wiki[i])
    
    return wiki


def remove_special_char_dataset(dataset):
    def remove(data):
        answer_start = data["answers"]["answer_start"][0]
        new_answer_start = len(remove_special_char(data["context"][:answer_start]))
        data["answers"]["answer_start"][0] = new_answer_start

        return {"context": remove_special_char(data["context"]), "answers": data["answers"]}
    
    return dataset.map(remove)
                               