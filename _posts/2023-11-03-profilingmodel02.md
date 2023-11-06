---
layout: single
title:  "Part 01. 온라인 메시지 작성자 프로파일링 모델 개발: 자연어 전처리 및 변수 개발"
categories: Project:Profileing_model
tag: [Feature Engineering, NLP]
---
<span style="color: #808080">#NLP #Feature Engineering #자연어 계량 #카카오톡 오픈채팅 대화 메시지</span>
<hr>

{: .notice--primary} 
💡**프로젝트 배경**<br>

개인정보 보호에 대한 사회적 분위기에 따라 구글 써드파티 제공 중단, 애플 사용자 정보 공개 중단 등 사용자 정보를 수집하는 것이 어려워지고 있음. 자사 플랫폼에 가입한 사용자 외 고객 데이터를 얻는 것은 더욱 어려움.<br><br> 
맥락정보를 활용한 고객 분석 및 타겟팅 전략에 대한 관심이 높아지고 있으나 대표적으로 고객이 생산하는 맥락정보인 채팅, 리뷰, 피드 등으로 파악할 수 있는 정보는 매우 제한적임. 특히, 사용자의 인구통계적 정보가 제공되는 경우는 매우 드물어 고객 분석 및 타겟팅에 활용할 수 없음.<br><br> 
**채팅, 리뷰, 피드 등에서 사용자의 인구 통계적 정보를 추정할 수 있다면, 고객 분석 및 타겟팅에 활용할 수 있을 뿐만아니라 챗봇/메타버스 서비스/CRM 서비스 등을 고도화시킬 수 있을 것으로 기대됨**<br><br>

{: .notice--primary} 
🎯**프로젝트 목적**<br>

고객이 온라인 상에서 생산하는 맥락정보(채팅, 리뷰, 피드 등의 텍스트 정보)에서 작성자의 인구 통계적 정보(성별/연령)를 추정하는 작성자 프로파일링 모델 개발<br><br>


## 변수 개발 요약<br>
**변수 개발의 필요성**
- 줄임말, 고유 표현, 자음 또는 모음으로만 구성된 불완전 표현, 문장기호, 특수기호, 이모지 등의 표현은 **기존의 NLP 전처리 및 언어모델의 Encoding 과정에서 제거됨**<br> 하지만, 이와 같은 표현들이 **발화자의 성별/연령에 대한 언어적 특징으로 보여지므로** 상관관계 파악 및 모델개발 시 반영을 위한 feauture 개발 필요
- 모든 표현을 encoding하는 방식은 비효율적일뿐만 아니라 사전 사전 학습된 모델에 적용할 수 없으므로 **표현의 특성을 계량화한 변수**가 요구됨
<br><br>

**인사이트/가설**
- 변형된 표현 및 특수표현 자체가 작성자의 인구 통계적 특성(성별/연령)과 중요한 상관관계를 가질 수 있음
1. Target feature(성별/연령)과 표현의 변형/특수표현 간의 상관관계 파악이 필요<br>
⇒ 상관관계 분석을 위한 변수 개발 필요

2. 변형된 표현 및 특수한 표현이 제거되지 않도록 전처리 수행
<br><br>

## 자연어 전처리


```python
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
```



    


- 결측값/중복값 제거


```python
import re

raw = df.shape[0]
print('- raw data: {:,}'.format(raw))


#결측값 제거
print('- null data: {:,}'.format(df['contents'].isnull().sum()))
df = df.dropna() 
deleted_null = df.shape[0]



#중복 제거
df = df.drop_duplicates() 
deleted_dup = df.shape[0]
print('- duplicated data: {:,}'.format(deleted_null - deleted_dup))



#문장 길이
df['contents_length'] = df['contents'].apply(lambda x : len(x))
```

    - raw data: 3,564,042
    - null data: 0
    - duplicated data: 0
    

### 주요 이모지 및 특수표현 표제화


```python
def emoji_lemmatization(sentence):
    heart_emoji = ['♡', '♥', '❤', '❤️', '🧡', '💛', '💚', '💙', '💜', '💕'] #
    star_emoji = ['☆', '★', '⭐', '🌟']
    kkk = ['𐨛', '𐌅', '⫬', 'ヲ', '刁', '㉪', 'ｦ']
    Period = ['ㆍ', 'ᆞ', 'ㆍ', '•', 'ᆢ']
    quote = ['”', '‘', '“']
    ect = [' ', 'ㅤ']
    
    for i in range(len(sentence)):
        if sentence[i] in heart_emoji:
            sentence = sentence.replace(sentence[i], '♥')
        elif sentence[i] in star_emoji:
            sentence = sentence.replace(sentence[i], '★')
        elif sentence[i] in kkk:
            sentence = sentence.replace(sentence[i], 'ㅋ')
        elif sentence[i] in Period:
            sentence = sentence.replace(sentence[i], '.')
        elif sentence[i] in quote:
            sentence = sentence.replace(sentence[i], '\'')
        elif sentence[i] in ect:
            sentence = sentence.replace(sentence[i], ' ')
        else:
            pass
    return(sentence)

def kkk_lemmatization(sentence):
    kkk2 =['ㅋ꙼̈', 'ㅋ̑̈', 'ㅋ̆̎', 'ㅋ̐̈', 'ㅋ̊̈', 'ㅋ̄̈', 'ㅋ̆̈', 'ㅋ̊̈', 'ㅋ̐̈', 'ㅋ̆̎']
    
    for i in range(len(kkk2)):
        if kkk2[i] in sentence:
            sentence =  sentence.replace(kkk2[i], 'ㅋ')
        else:
            pass
    return(sentence)

text_sentence = '❤🧡💛테스트💚💙💜☆입니다★ᆢ⭐ ヲ𐨛𐌅⫬ㅋ̄̈ㅋ꙼̈ㅋ̆̎ㅋ̐̈ㅋ̊̈ㅋ̄̈ㅋ꙼̈ㅋ̆̎ㅋ̐̈ㅋ̊̈'
text_sentence = emoji_lemmatization(text_sentence)
text_sentence = kkk_lemmatization(text_sentence)
text_sentence
```




    '♥♥♥테스트♥♥♥★입니다★.★ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'




```python
df['contents'] = df['contents'].progress_apply(emoji_lemmatization)
df['contents'] = df['contents'].progress_apply(kkk_lemmatization)
```

    100%|████████████████████████████████████████████████████████████████████████████| 3564042/3564042 [05:13<00:00, 11354.18it/s]
    100%|███████████████████████████████████████████████████████████████████████████| 3564042/3564042 [00:10<00:00, 330031.43it/s]
    

### 반복되는 동일 음절 표제화 처리


```python
def duplicated_spelling_reduction(sentence):
    reduced_spellings = []
    duplicated_num = 1
    for i in range(len(sentence)):
        spelling = sentence[i]
        try:
            previous_spelling = sentence[i-1]
            
        except:
            previous_spelling = 'first_spelling'
        
        if spelling == previous_spelling:
            duplicated_num += 1
        else:
            duplicated_num = 1
            pass
        
        if duplicated_num <= 5:
            reduced_spellings.append(spelling)
        else:
            pass      
        
    reduced_sentence = ''.join(reduced_spellings).replace('   ', ' ').replace('  ', ' ')
    return(reduced_sentence)

text_sentence = '    안녕안녕 헤헤헤 ㅋㅋㅋㅋ ㅎㅎㅎㅎㅎㅎㅎㅎ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'
duplicated_spelling_reduction(text_sentence)
```




    ' 안녕안녕 헤헤헤 ㅋㅋㅋㅋ ㅎㅎㅎㅎㅎ ㅋㅋㅋㅋㅋ'




```python
df['contents'] = df['contents'].progress_apply(duplicated_spelling_reduction)
```

    100%|████████████████████████████████████████████████████████████████████████████| 3564042/3564042 [02:02<00:00, 29125.25it/s]
    

### 공백(중복 띄어쓰기)/5어절 미만 문장 제거


```python
#띄어쓰기 문장 제거
df['contents'] = df['contents'].apply(lambda x : re.sub(r'\s', ' ', x))  #띄어쓰기 중복 ''로 변경
mask = df['contents'].isin([' '])
df = df[~mask].reset_index(drop = True) 
deleted_white = df.shape[0]
white = deleted_dup - deleted_white
print("- white space data: {:,}({}%)".format(white, round(white/deleted_dup*100, 2)))

#어절 카운팅
df['word_bunch'] = df['contents'].apply(lambda x: len(x.split(' ')))


#5어절 미만 문장 제거
cutoff = df.loc[df['word_bunch'] < 5].shape[0] 
df = df.loc[df['word_bunch'] >= 5] 
deleted_cutoff = df.shape[0]
print("- cutoff data: {:,}({}%)".format(cutoff, round(cutoff/deleted_white*100, 2)))

print("- total processed data: {:,}".format(deleted_cutoff))
```

    - white space data: 0(0.0%)
    - cutoff data: 61,130(1.72%)
    - total processed data: 3,502,912



```python
df.to_csv('SNS_FULL_Dataset(텍스트 전처리).csv', index=False)
```

