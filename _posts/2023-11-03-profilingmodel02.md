---
layout: single
title:  "Part 01. 온라인 메시지 작성자 프로파일링 모델 개발: 자연어 전처리 및 변수개발"
categories: Project:Profileing_model
tag: [NLP, 불용어처리, 표제화, 토크니제이션, 변수개발]
---
<span style="color: #808080">#NLP #불용어 처리 #표제화 #토크니제이션 #변수개발</span>
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
- 줄임말, 고유 표현, 자음 또는 모음으로만 구성된 불완전 표현, 문장기호, 특수기호, 이모지 등의 표현은 **기존의 NLP 전처리 및 언어모델의 Encoding 과정에서 제거됨**. 하지만, 이와 같은 표현들이 **발화자의 성별/연령에 대한 언어적 특징으로 보여지므로** 상관관계 파악 및 모델개발 시 반영을 위한 feauture 개발 필요
- 모든 표현을 encoding하는 방식은 비효율적일뿐만 아니라 사전 사전 학습된 모델에 적용할 수 없으므로 **표현의 특성을 계량화한 변수**가 요구됨
<br><br>

### 1. 표현의 길이와 철자의 다양성을 바탕으로 한 복잡도 계산
- **Complexity of Specific Expression(C, 특수표현 복잡도)**: **표현 t가 복잡한 정도**를 설명함
    > 전처리에서 대부분 소실되는 자음 또는 모음으로 구성된 표현, 문장기호, 특수기호, 이모지로 구성된 표현에 대한 복잡도를 계산<br>
    > 자음 또는 모음으로 구성된 표현을 한 분류로 하고, 문장기호/특수기호/이모지를 한 분류로 구분하여 계산(형태소 분류 및 정규식 처리에 용이)
    > <br><br>
    > $C_i= ln[ 1/Π(U_i / L_i) * L_i]$
    > - $U_i$ : 표현 $t_i$ 가 포함하는 고유한 철자 또는 기호의 종류 수
    > - $L_i$ : 표현 $t_i$ 의 길이(철자 및 기호 개수)
    > <br>
    > 표현을 이루는 고유 철자 및 기호 별 점유율 $U_i / L_i$ 을 모두 곱하고, 표현의 길이를 곱하여 고유 철자의 종류와 표현의 길이에 비례하는 복잡도 측정. 다양한 철자가 사용될 수록, 사용된 고유 철자 및 기호 수가 많을 수록 복잡도가 커짐
&nbsp;&nbsp;⇒ 전처리 전의 raw 텍스트에 대해 측정하며, 전처리에서 소실되는 특수표현들의 정보를 일부 반영함
<br><br>

### 2. 각 표현의 종속변수(성별/연령)에 대한 Odds Ratio를 계산
- **Relative Bias(RB, 상대 편향도)**: 표현 t가 등장 했을 때, 텍스트의 **작성자가 특정 성별/연령인 정도**를 설명함
    > **Relative Bias of Gender**(RBG, 상별에 대한 상대 편향도)
    > : 표현 $t_i$ 가 등장한 문장 $s$ 의 작성자 성별이 남자 $male$ 또는 여자 $female$ 인 정도<br>
    > <br>
    > $RBG_i = ln[ p( s_{male} ⏐ t_i∈s_{male} ) / p( s_{female} ⏐ t_i∈s_{female} ) ]$
    > - $t_i$ : 문서 내 i번 째 표현<br>
    > - $s_{male}$ : 작성자의 성별이 남성(m)인 문장
    > - $s_{female}$ : 작성자의 성별이 여성(f)인 문장
    > <br>
    > 작성자의 성별이 남성인 문장 $s_{male}$ 에서 표현 $t_i$ 가 등장한 비율 ÷ 여성인 문장 $s_{female}$ 에서 표현 $t_i$ 가 등장한 비율, 0~1 사이의 Skewed한 값을 가짐으로 log를 취해 정규화
    > ⇒ $t_i$ 가 등장했을 때 작성자의 성별이 남자 $m$ 또는 여자 $f$ 인 정도
    <br>    

    > **Relative Bias of Age**(RBG, 연령에 대한 상대 편향도)<br>
    > : 표현 $t_i$ 가 등장한 문장 $s$ 의 작성자 연령이 특정 연령대 $age$ 인 정도
    > &nbsp;&nbsp;연령대는 20대 미만/20대/30대/40대/50대 이상 5 class로 분류<br>
    > <br>
    > $RBAi = ln[ p( s_{age} ⏐ t_i∈s_{age} ) / p( s_{other} ⏐ t_i∈s_{other} ) ]$
    > - $t_i$ : 문서 내 i번 째 표현
    > - $s_{age}$ : 작성자의 연령대가 age인 문장
    > - $s_{other}$ : 작성자의 연령대가 age가 아닌 문장
    > <br>
    > 작성자의 연령대가 20대인 문장 $s_{a20}$ 에서 표현 $t_i$ 가 등장한 비율 ÷ 연령대가 20대가 아닌 문장 $s_{other}$ 에서 표현 $t_i$ 가 등장한 비율, 0~1 사이의 Skewed한 값을 가짐으로 log를 취해 정규화
    > ⇒ $t_i$가 등장했을 때 작성자의 연령대가 age인 정도
    <br>  
    
- **Relative frequency (RF, 상대 빈출도)**: 텍스트의 작성자가 특정 성별/연령일 때 타 성별/연령 보다 **표현 t를 상대적으로 많이 사용하는 정도**를 설명함
    > 상대 편형도는 문장에서 표현의 출연여부를 기준으로 하는 반면, 상대 빈출도는 빈도수를 기준으로 함
    > 상대 편향도와 계산 방식이 유사하므로 설명은 생략함<br>
    > <br>
    > $RFG_i = ln[ p( t_i ⏐ s_{male} ) / p( t_i ⏐ s_{female} ) ]$
    > $RFAi = ln[ p( t_i ⏐ s_{age} ) / p( t_i ⏐ s_{other} ) ]$
    <br>
    
- 문장이 포함하는 토큰 전체에 대한 상대 편향도 및 상대 빈출도의 통계치를 계산함으로써 기존의 NLP 방식에서 소실되는 토큰 정보의 일부분 보완할 수 있음
- 표현별 상대 편향도 및 상대 편향도에 대한 딕셔너리 구축이 필요
- Odds ratio의 값을 일반화할 수 있을 만큼 충분한 데이터가 전제되어야 하며, 본 데이터셋은 그에 준한다고 가정함
- 상대 편향도와 상대 빈출도가 매우 유사함으로 유의성 검토를 통해 1가지만 채택
<br> 




## 01. 특수표현 복잡도 측정
- 불용어 처리/표제화 전에 수행
- 자음 또는 모음으로 구성된 불완전 표현 / 문장기호, 특수기호, 이모지 표현을 구분하여 측정
- 문장에서 출현하는 특수표현의 개수, 평균, 표준편차, 최대값, 최소값을 계산




```python
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import re
import numpy as np
from collections import Counter

df = pd.read_csv("SNS_FULL_Dataset(raw_중복된 메세지 제거).csv")
```



<br><br>
### 자음 또는 모음으로 구성된 불완전 표현의 복잡도


```python
def spell_complexity(sentence):
    sequences = re.findall(r'[ㄱ-ㅎㅏ-ㅣ]+', sentence)
    specific_sequences = [seq for seq in sequences if len(seq) > 0]
    
    N = len(specific_sequences)
    
    if N != 0:
        complexity_list=[]
        for seq in specific_sequences:
            spell_list = []
            for s in seq:
                spell_list.append(s)  
```




```python
print(spell_complexity('안녕 ㅠㅠ ㅠㅠㅠ ㅠㅜ ㅇㅋㅇㅋ ㄷㅋ-ㅁㅁ- ㅠ'))
```

    (7, 1.3451969221982076, 0.9131088298012547, 2.772588722239781, 0.0)




```python
print(spell_complexity('안 ㅠㅠ ㅋㅋㅋ'))
```

    (2, 0.8958797346140275, 0.20273255405408225, 1.0986122886681098, 0.6931471805599453)



<br><br> 
### 문장기호, 특수기호, 이모지 표현에 대한 복잡도




```python
def symbol_complexity(sentence):
    sequences = re.findall(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]+', sentence)
    specific_sequences = [seq.strip() for seq in sequences if seq.strip()]
    
    N = len(specific_sequences)
    
    if N != 0:
        complexity_list=[]
        for seq in specific_sequences:
            spell_list = []
            for s in seq:
                spell_list.append(s)  

            uniqe_list = Counter(spell_list)
            uniqe_freq = list(uniqe_list.values())

            L = len(spell_list)
            uniqe_num =len(uniqe_freq)


            ratio_list = []
            for fu in uniqe_freq:
                ratio = (fu/L)
                ratio_list.append(ratio)

            ans = 1
            for r in ratio_list:
                ans *= r

            complexity = np.log(1/ans*L)
            complexity_list.append(complexity)
            
        mean = np.mean(complexity_list)
        std =  np.std(complexity_list)
        max_ = np.max(complexity_list)
        min_ = np.min(complexity_list)
    else:
        mean = 0
        std = 0
        max_ = 0
        min_ = 0
        
    return N, mean, std, max_, min_
```


```python
print(symbol_complexity('ㅠㅠ 안녕! ??... 예시: 2^^; -_-;; o_O 😘 🙏 ☺️ ❤️🧡💛'))
```

    (10, 1.970161458941443, 2.3466123014783795, 6.931471805599453, 0.0)



```python
print(symbol_complexity('ㅠㅠ 안녕😘😘😘 😘😘'))
```

    (2, 0.8958797346140275, 0.20273255405408225, 1.0986122886681098, 0.6931471805599453)



```python
#데이터프레임 컬럼값으로 변환
df['spell_complexity'] = df['contents'].progress_apply(lambda x:spell_complexity(str(x)))
df['spell_num'] = df['spell_complexity'].progress_apply(lambda x:x[0])
df['spell_mean'] = df['spell_complexity'].progress_apply(lambda x:x[1])
df['spell_std'] = df['spell_complexity'].progress_apply(lambda x:x[2])
df['spell_max'] = df['spell_complexity'].progress_apply(lambda x:x[3])
df['spell_min'] = df['spell_complexity'].progress_apply(lambda x:x[4])


df['symbol_complexity'] = df['contents'].progress_apply(lambda x:symbol_complexity(str(x)))
df['symbol_num'] = df['symbol_complexity'].progress_apply(lambda x:x[0])
df['symbol_mean'] = df['symbol_complexity'].progress_apply(lambda x:x[1])
df['symbol_std'] = df['symbol_complexity'].progress_apply(lambda x:x[2])
df['symbol_max'] = df['symbol_complexity'].progress_apply(lambda x:x[3])
df['symbol_min'] = df['symbol_complexity'].progress_apply(lambda x:x[4])


df = df.drop(columns={'spell_complexity', 'symbol_complexity'})
df = df.fillna(0)
df
```

    100%|████████████████████████████████████████████████████████████████████████████| 3564042/3564042 [04:53<00:00, 12152.45it/s]
    ...생략...





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
      <th>sex</th>
      <th>age</th>
      <th>resident</th>
      <th>contents</th>
      <th>length</th>
      <th>spell_num</th>
      <th>spell_mean</th>
      <th>spell_std</th>
      <th>spell_max</th>
      <th>spell_min</th>
      <th>symbol_num</th>
      <th>symbol_mean</th>
      <th>symbol_std</th>
      <th>symbol_max</th>
      <th>symbol_min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>나지금밥머거2시간걸어서 번화가찾았어..ㅜㅜ 잉ㅜㅜ ㅎㅎㅎㅎ오좋겠네 ㅋㄱㅋㄱㄱㄱㄱ아니...</td>
      <td>127</td>
      <td>6</td>
      <td>2.185021</td>
      <td>1.299761</td>
      <td>3.765840</td>
      <td>0.693147</td>
      <td>4</td>
      <td>0.173287</td>
      <td>0.300142</td>
      <td>0.693147</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>개인및관계</td>
      <td>남성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>헐 ㅠㅠ 언넝호텔들가ㅠㅠ 엄청피건할첸데 나는인낫러요 나 두시출근이다ㅎㅎㅎㅎ 퀵으로한...</td>
      <td>130</td>
      <td>6</td>
      <td>0.961387</td>
      <td>0.555163</td>
      <td>1.609438</td>
      <td>0.000000</td>
      <td>6</td>
      <td>2.041180</td>
      <td>1.296771</td>
      <td>4.212128</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>학생이면좋구! 왜혼자다니냐고오..... 와 내친군학교나감 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 그르네 ...</td>
      <td>56</td>
      <td>1</td>
      <td>2.197225</td>
      <td>0.000000</td>
      <td>2.197225</td>
      <td>2.197225</td>
      <td>3</td>
      <td>1.404043</td>
      <td>1.072424</td>
      <td>2.602690</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>개인및관계</td>
      <td>남성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>훔 학생 없는데...주변에... 아니 복학하고 학교를 못가는데 어케 친구가있냐.. ...</td>
      <td>74</td>
      <td>1</td>
      <td>2.079442</td>
      <td>0.000000</td>
      <td>2.079442</td>
      <td>2.079442</td>
      <td>4</td>
      <td>0.997246</td>
      <td>0.175572</td>
      <td>1.098612</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>30대</td>
      <td>충청북도</td>
      <td>참나 내가뭐얼마나그랬다고 웃기는사람이야지짜 너무화난당.. 근데오빠는말을또 잘해서 내...</td>
      <td>146</td>
      <td>2</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>3564042 rows × 16 columns</p>

</div>




```python
df.to_csv('SNS_FULL_Dataset(raw_표현 복잡도 계산).csv', index=False)
```



<br><br>
## 02. 자연어 전처리
- 3번 이상 반복되는 동일 음절은 3음절로 표제화 *ex. ㅇㅇㅇㅇㅇ ⇒ ㅇㅇㅇ
- 문장기호, 특수기호, 이모지, 특수폰트 표현 표제화 *ex. ❤🧡💛 ⇒ ♥️♥️♥️ / ヲ𐨛𐌅⫬ ⇒ ㅋㅋㅋㅋ
- 연속된 띄어쓰기(공백) 한 번으로 처리
- 5어절 미만 말뭉치 데이터 제거 


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



#문장 길이 컬럼 추가가
df['contents_length'] = df['contents'].apply(lambda x : len(x))
```

    - raw data: 3,564,042
    - null data: 0
    - duplicated data: 0
    
<br><br>
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
    
<br><br>
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
    
<br><br>
### 연속된 띄어쓰기 처리/5어절 미만 문장 제거


```python
#연속되 띄어쓰기 처리
df['contents'] = df['contents'].apply(lambda x : re.sub(r'\s', ' ', x))  #띄어쓰기 중복 ''로 변경
mask = df['contents'].isin([' '])
df = df[~mask].reset_index(drop = True) 
deleted_white = df.shape[0]
white = deleted_dup - deleted_white
print("- white space data: {:,}({}%)".format(white, round(white/deleted_dup*100, 2)))
```

    - white space data: 0(0.0%)


```python
#5어절 미만 문장 제거
df['word_bunch'] = df['contents'].apply(lambda x: len(x.split(' '))) # 어절 카운팅

cutoff = df.loc[df['word_bunch'] < 5].shape[0] 
df = df.loc[df['word_bunch'] >= 5] 
deleted_cutoff = df.shape[0]
print("- cutoff data: {:,}({}%)".format(cutoff, round(cutoff/deleted_white*100, 2)))
```

    - cutoff data: 61,130(1.72%)



```python
df.to_csv('SNS_FULL_Dataset(텍스트 전처리).csv', index=False)
```

