---
layout: single
title:  "Part 02. 온라인 메시지 작성자 프로파일링 모델 개발: 전처리 및 변수개발"
categories: Project:프로파일링_모델_개발
tag: [NLP, 불용어처리, 표제화, 토크나이즈, 변수개발]
---
<span style="color: #808080">#NLP #불용어 처리 #표제화 #토크나이즈 #변수개발 #자연어 계량</span>
<hr>

{: .notice--primary} 
💡**프로젝트 배경**<br>

개인정보 보호에 대한 사회적 분위기에 따라 구글 써드파티 제공 중단, 애플 사용자 정보 공개 중단 등 사용자 정보를 수집하는 것이 어려워지고 있음. 자사 플랫폼에 가입한 사용자 외 고객 데이터를 얻는 것은 더욱 어려움.<br><br> 
맥락정보를 활용한 고객 분석 및 타겟팅 전략에 대한 관심이 높아지고 있으나 대표적으로 고객이 생산하는 맥락정보인 채팅, 리뷰, 피드 등으로 파악할 수 있는 정보는 매우 제한적임. 특히, 사용자의 인구통계적 정보가 제공되는 경우는 매우 드물어 고객 분석 및 타겟팅에 활용할 수 없음.<br><br> 
**채팅, 리뷰, 피드 등에서 사용자의 인구 통계적 정보를 추정할 수 있다면, 고객 분석 및 타겟팅에 활용할 수 있을 뿐만아니라 챗봇/메타버스 서비스/CRM 서비스 등을 고도화시킬 수 있을 것으로 기대됨**<br><br>
 
{: .notice--primary} 
🎯**프로젝트 목적**<br>

고객이 온라인 상에서 생산하는 맥락정보(채팅, 리뷰, 피드 등의 텍스트 정보)에서 **작성자의 인구 통계적 정보(성별/연령)를 추정하는 작성자 프로파일링 모델 개발**<br><br>
 
 
## 변수 개발 요약<br>
**변수 개발의 필요성**
- 줄임말, 고유 표현, 자음 또는 모음으로만 구성된 불완전 표현, 문장기호, 특수기호, 이모지 등의 표현은 **기존의 NLP 전처리 및 언어모델의 Encoding 과정에서 제거됨**. 하지만, 이와 같은 표현들이 **발화자의 성별/연령에 대한 언어적 특징으로 보여지므로** 상관관계 파악 및 모델개발 시 반영을 위한 feauture 개발 필요
- 모든 표현을 encoding하는 방식은 비효율적일뿐만 아니라 사전 사전 학습된 모델에 적용할 수 없으므로 **표현의 특성을 계량화한 변수**가 요구됨
<br>

 
### 1. 표현의 길이와 철자의 다양성을 바탕으로 한 복잡도 계산
**Complexity of Specific Expression(C, 특수표현 복잡도)** 
- 표현 t가 복잡한 정도를 설명함
- 전처리에서 대부분 소실되는 자음 또는 모음으로 구성된 표현, 문장기호, 특수기호, 이모지로 구성된 표현에 대한 복잡도를 계산
- 자음 또는 모음으로 구성된 표현을 한 분류로 하고, 문장기호/특수기호/이모지를 한 분류로 구분하여 계산(형태소 분류 및 정규식 처리에 용이)
> $C_i= ln[ 1/Π(U_i / L_i) * L_i]$
> $U_i$ : 표현 $t_i$ 가 포함하는 고유한 철자 또는 기호의 종류 수
> $L_i$ : 표현 $t_i$ 의 길이(철자 및 기호 개수)
> <br><br>
> 표현을 이루는 고유 철자 및 기호 별 점유율 $U_i / L_i$ 을 모두 곱하고, 표현의 길이를 곱하여 고유 철자의 종류와 표현의 길이에 비례하는 복잡도 측정. 다양한 철자가 사용될 수록, 사용된 고유 철자 및 기호 수가 많을 수록 복잡도가 커짐<br>
⇒ 전처리 전의 raw 텍스트에 대해 측정하며, 전처리에서 소실되는 특수표현들의 정보를 일부 반영함

<br><br>
### 2. 각 표현의 종속변수(성별/연령)에 대한 Odds Ratio를 계산
**Relative Bias(RB, 상대 편향도)**: 표현 t가 등장 했을 때, 텍스트의 **작성자가 특정 성별/연령인 정도**를 설명함
<br><br>
a. **Relative Bias of Gender**(RBG, 상별에 대한 상대 편향도)<br>
- 표현 $t_i$ 가 등장한 문장 $s$ 의 작성자 성별이 남자 $male$ 또는 여자 $female$ 인 정도
> $RBG_i = ln[ p( s_{male} ⏐ t_i∈s_{male} ) / p( s_{female} ⏐ t_i∈s_{female} ) ]$
>  $t_i$ : 문서 내 i번 째 표현<br>
> $s_{male}$ : 작성자의 성별이 남성(m)인 문장
>  $s_{female}$ : 작성자의 성별이 여성(f)인 문장
> <br><br>
> 작성자의 성별이 남성인 문장 $s_{male}$ 에서 표현 $t_i$ 가 등장한 비율 ÷ 여성인 문장 $s_{female}$ 에서 표현 $t_i$ 가 등장한 비율, 0~1 사이의 Skewed한 값을 가짐으로 log를 취해 정규화<br>
  ⇒ $t_i$ 가 등장했을 때 작성자의 성별이 남자 $m$ 또는 여자 $f$ 인 정도

<br><br>
 b. **Relative Bias of Age**(RBG, 연령에 대한 상대 편향도)<br>
  - 표현 $t_i$ 가 등장한 문장 $s$ 의 작성자 연령이 특정 연령대 $age$ 인 정도를 설명함<br>
  - 연령대는 20대 미만/20대/30대/40대/50대 이상 5 class로 분류
> $RBAi = ln[ p( s_{age} ⏐ t_i∈s_{age} ) / p( s_{other} ⏐ t_i∈s_{other} ) ]$
> $t_i$ : 문서 내 i번 째 표현
> $s_{age}$ : 작성자의 연령대가 age인 문장
> $s_{other}$ : 작성자의 연령대가 age가 아닌 문장
> <br><br>
> 작성자의 연령대가 20대인 문장 $s_{a20}$ 에서 표현 $t_i$ 가 등장한 비율 ÷ 연령대가 20대가 아닌 문장 $s_{other}$ 에서 표현 $t_i$ 가 등장한 비율, 0~1 사이의 Skewed한 값을 가짐으로 log를 취해 정규화<br>
⇒ $t_i$가 등장했을 때 작성자의 연령대가 age인 정도<br>
<br>

 
**Relative frequency (RF, 상대 빈출도)**
- 텍스트의 작성자가 특정 성별/연령일 때 타 성별/연령 보다 **표현 t를 상대적으로 많이 사용하는 정도**를 설명함
- 상대 편형도는 문장에서 표현의 출연여부를 기준으로 하는 반면, 상대 빈출도는 빈도수를 기준으로 함<br>
- 상대 편향도와 계산 방식이 유사하므로 설명은 생략함
>  $RFG_i = ln[ p( t_i ⏐ s_{male} ) / p( t_i ⏐ s_{female} ) ]$
>  $RFAi = ln[ p( t_i ⏐ s_{age} ) / p( t_i ⏐ s_{other} ) ]$


<br><br>
- 문장이 포함하는 토큰 전체에 대한 상대 편향도 및 상대 빈출도의 통계치를 계산함으로써 기존의 NLP 방식에서 소실되는 토큰 정보의 일부분 보완할 수 있음
- 표현별 상대 편향도 및 상대 편향도에 대한 딕셔너리 구축이 필요
- Odds ratio의 값을 일반화할 수 있을 만큼 충분한 데이터가 전제되어야 하며, 본 데이터셋은 그에 준한다고 가정함
- 상대 편향도와 상대 빈출도가 매우 유사함으로 유의성 검토를 통해 1가지만 채택
<br>

<br><br>
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
df['contents'] = df['contents'].apply(lambda x : re.sub(r'\s', ' ', x))  # 연속된 띄어쓰기를 띄어쓰기 1칸으로 처리
mask = df['contents'].isin([' ']) # 띄어쓰기만 존제하는 데이터 제거
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


<br><br>
### 토크나이즈
- konlpy 라이브러리의 Okt를 사용하여 형태소 분리
- 동일 철자이나 다른 형태소를 갖는 토큰을 파악하기 위해 '토큰(형태소)' 형식으로 토크나이즈즈

```python
def pos_tokenizer(sentence):    #POS 기준 토크나이제이션
    import konlpy
    okt = konlpy.tag.Okt() 
    pos_list = okt.pos(str(sentence)) #토큰/형태소 튜플처리

    token_arry = []
    for t, m in pos_list:
        token = '{}({})'.format(t, m)
        token_arry.append(token)
    
    token_count = len(token_arry)
    tokenized_sentence = ', '.join(token_arry)
    return tokenized_sentence, token_count
```


```python
# 데이터가 커서 분리하여 처리
df1 = df[:600000]
df2 = df[600000:1200000]
df3 = df[1200000:1800000]
df4 = df[1800000:2400000]
df5 = df[2400000:3000000]
df6 = df[3000000:]

df_li = [df1, df2, df3, df4, df5, df6]

for d in df_li:
    d['tokenized'] = d['contents'].progress_apply(pos_tokenizer)
    d['tokenized'] = d['tokenized'].progress_apply(lambda x : x[0])
    d['token_count'] = d['tokenized'].progress_apply(lambda x: x[1])
    
tokenized_df = pd.concat(df_li, ignore_index=True)
tokenized_df = tokenized_df.drop(columns={'tokenized'})
tokenized_df[['sex', 'age', 'contents', 'tokenized', 'token_count']]
```

    100%|████████████████████████████████████████████████████████████████████████████████| 600000/600000 [36:11<00:00, 276.37it/s]
    ...생략...





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>age</th>
      <th>contents</th>
      <th>tokenized</th>
      <th>token_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>여성</td>
      <td>20대</td>
      <td>나지금밥머거2시간걸어서 번화가찾았어..ㅜㅜ 잉ㅜㅜ ㅎㅎㅎㅎ오좋겠네 ㅋㄱㅋㄱㄱㄱㄱ아니...</td>
      <td>나(Noun), 지금(Noun), 밥(Noun), 머거(Verb), 2시간(Numb...</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>남성</td>
      <td>20대</td>
      <td>헐 ㅠㅠ 언넝호텔들가ㅠㅠ 엄청피건할첸데 나는인낫러요 나 두시출근이다ㅎㅎㅎㅎ 퀵으로한...</td>
      <td>헐(Verb), ㅠㅠ(KoreanParticle), 언(Modifier), 넝(No...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>여성</td>
      <td>20대</td>
      <td>학생이면좋구! 왜혼자다니냐고오..... 와 내친군학교나감 ㅋㅋㅋㅋㅋ 그르네 막졸업한...</td>
      <td>학생(Noun), 이(Suffix), 면(Josa), 좋구(Adjective), !...</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남성</td>
      <td>20대</td>
      <td>훔 학생 없는데...주변에... 아니 복학하고 학교를 못가는데 어케 친구가있냐.. ...</td>
      <td>훔(Noun), 학생(Noun), 없는데(Adjective), ...(Punctua...</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>여성</td>
      <td>30대</td>
      <td>참나 내가뭐얼마나그랬다고 웃기는사람이야지짜 너무화난당.. 근데오빠는말을또 잘해서 내...</td>
      <td>참나(Noun), 내(Noun), 가(Josa), 뭐(Noun), 얼마나(Noun)...</td>
      <td>63</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>3502912 rows × 20 columns</p>
</div>




```python
tokenized_df.to_csv('350만_Tokenized.csv', index = False)
```


<br><br>
### 토큰 딕셔너리 구축 및 스크리닝
**딕셔너리 구축**
- 빈도 0.0001 이상의 토큰만 사용<br>
[토큰 딕셔너리(구글 스프레드)](https://docs.google.com/spreadsheets/d/1wKv4hAfJD_ToORv1Q7QGWsCjYUQTZKHRzZMNCxVPvZ4/edit?usp=sharing)
https://docs.google.com/spreadsheets/d<br>/1wKv4hAfJD_ToORv1Q7QGWsCjYUQTZKHRzZMNCxVPvZ4/edit?usp=sharing

**스크리닝 결과**
- Okt 딕셔너리에 포함되지 않은 표현이 파괴됨<br>
  ⇒ 두드러지는 표현 교정(분리된 토큰 결합)

       
- 분류가 모호하거나 부정확한 형태소 파악
  - 의미적으로 해석이 어려운 Eomi, VerbPrefix, Suffix<br>
    ⇒ 전/후 토큰과 병합 및 형태소 재분류 
  - Modifier,Determiner 두 형태소의 개념적 구분이 명확하지 않음<br>
    ⇒ 두 형태소 병합 
  - Foreign 분류가 부정확(특수기호 및 기타 표현 포함)<br>
    ⇒ 형태소별 재분류 



```python
#메모리아웃 되는 것을 방지하기 위해 데이터를 일정 크기 이하로 Split
def split_dataframe(dataframe):
    total_length = len(dataframe)
    splited_li = []  # splited_li는 반드시 초기화되어야 합니다.

    if len(dataframe) > 100000:
        split_size = 100000
        num_split = total_length // split_size + 1

        for i in range(num_split):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size
            try:
                splited = dataframe[start_idx:end_idx]
            except:
                splited = dataframe[start_idx:]

            # 작은 그룹을 리스트에 추가
            splited_li.append(splited)
    else:
        splited_li.append(dataframe)
    return splited_li
```



```python
def occur_countor(df):
    total_length = len(df)
    splited_li = split_dataframe(df)

    Occur_dict = {}
    for splited_df in splited_li: 
        merge_list = []
        for index, row in tqdm(splited_df.iterrows(), total=len(splited_df), desc="전체 토큰 병합", mininterval=0.1):
            token_list = row['tokenized'].split(', ')
            for token in token_list:
                merge_list.append(token)

        count_list = Counter(merge_list).most_common()

        sort_arry = []
        for d, c in tqdm(count_list, total=len(count_list), desc="빈도순 정렬", mininterval=0.1):
            for i in range(c):
                sort_arry.append(d)

        unique_dic = Counter(sort_arry)
        unique_token = list(unique_dic.keys())
        unique_freq = list(unique_dic.values())
        
        for key, value in zip(unique_token, unique_freq):  # zip을 사용하여 두 리스트를 동시에 순회
            if key in Occur_dict:
                Occur_dict[key] += value
            else:
                Occur_dict[key] = value

    token_li = Occur_dict.keys()
    freq_li = Occur_dict.values()

    Occur_dic = pd.DataFrame({'Token': token_li, 'Token_freq': freq_li})
    Occur_dic = Occur_dic.sort_values(by=['Token_freq'], ascending=False).reset_index(drop=True) 
    Occur_dic['Total_ratio'] = Occur_dic['Token_freq'].apply(lambda x: x/total_length)
    Occur_dic = Occur_dic.loc[Occur_dic['Total_ratio'] >= 0.0001]
    return Occur_dic
```




```python
Occur_dic['word'] = Occur_dic['Token'].apply(lambda x: ''.join(x.split('(')[:-1]))
Occur_dic['pos'] = Occur_dic['Token'].apply(lambda x: x.split('(')[-1].replace(')', ''))
Occur_dic = Occur_dic[['Token', 'word', 'pos', 'Token_freq', 'Total_ratio']]
Occur_dic
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Token</th>
      <th>word</th>
      <th>pos</th>
      <th>Token_freq</th>
      <th>Total_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>?(Punctuation)</td>
      <td>?</td>
      <td>Punctuation</td>
      <td>1865214</td>
      <td>0.532475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ㅋㅋㅋㅋㅋ(KoreanParticle)</td>
      <td>ㅋㅋㅋㅋㅋ</td>
      <td>KoreanParticle</td>
      <td>1526323</td>
      <td>0.435730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>에(Josa)</td>
      <td>에</td>
      <td>Josa</td>
      <td>1307922</td>
      <td>0.373381</td>
    </tr>
    <tr>
      <th>3</th>
      <td>가(Josa)</td>
      <td>가</td>
      <td>Josa</td>
      <td>1142959</td>
      <td>0.326288</td>
    </tr>
    <tr>
      <th>4</th>
      <td>이(Josa)</td>
      <td>이</td>
      <td>Josa</td>
      <td>913782</td>
      <td>0.260864</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18142</th>
      <td>오는구나(Verb)</td>
      <td>오는구나</td>
      <td>Verb</td>
      <td>351</td>
      <td>0.000100</td>
    </tr>
    <tr>
      <th>18143</th>
      <td>the(Alpha)</td>
      <td>the</td>
      <td>Alpha</td>
      <td>351</td>
      <td>0.000100</td>
    </tr>
    <tr>
      <th>18144</th>
      <td>문상(Noun)</td>
      <td>문상</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
    </tr>
    <tr>
      <th>18145</th>
      <td>앚(Noun)</td>
      <td>앚</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
    </tr>
    <tr>
      <th>18146</th>
      <td>센트럴(Noun)</td>
      <td>센트럴</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
    </tr>
  </tbody>
</table>
<p>18147 rows × 5 columns</p>



```python
Occur_dic.to_csv('350만_Occur_dic.csv', index=False)
```




<br><br>
### 주요 오분류  형태소 재분류


```python
# 오분류되는 주요 토큰 형태소 재분류
def fix_foreign(tokenized):
    missing_Adverb = [', 후에(Foreign)', ', 후(Foreign)', ', 초에(Foreign)', ', 쯤에(Foreign)',
                      ', 쯤(Foreign)', ', 정도에(Foreign)', ', 정도(Foreign)', ', 전에는(Foreign)',
                      ', 전에(Foreign)', ', 전(Foreign)', ', 이상(Foreign)', ', 안에(Foreign)',
                      ', 부터는(Foreign)', ', 부터(Foreign)', ', 반쯤(Foreign)', ', 반에(Foreign)',
                      ', 반부터(Foreign)', ', 반까지(Foreign)', ', 반(Foreign)', ', 밖에(Foreign)',
                      ', 말에(Foreign)', ', 말(Foreign)', ', 만에(Foreign)', ', 만(Foreign)',
                      ', 마다(Foreign)', ', 뒤에(Foreign)', ', 뒤(Foreign)', ', 동안(Foreign)',
                    
                      ', 넘어서(Foreign)', ', 넘게(Foreign)', ', 남음(Foreign)', ', 꺼(Foreign)',
                      ', 까진데(Foreign)', ', 까진(Foreign)', ', 까지야(Foreign)', ', 까지만(Foreign)',
                      ', 까지는(Foreign)', ', 까지(Foreign)', ', 간(Foreign)', ', 이후(Foreign)',
                      ', 이후에(Foreign)', ', 내로(Foreign)', ', 경(Foreign)', ', 말까지(Foreign)',
                      ', 전까지(Foreign)', ', 중에(Foreign)', ', 즘(Foreign)', ', 내내(Foreign)',
                      ', 정도는(Foreign)', ', 초(Foreign)', ', 얼마(Foreign)', ', 정도면(Foreign)',
                      ', 이내(Foreign)', ', 내(Foreign)', ', 간의(Foreign)', ', 간은(Foreign)',
                      ', 약(Foreign)', ', 보다(Foreign)', ', 전엔(Foreign)', ', 까지니까(Foreign)',
                      ', 정도만(Foreign)', ', 사이에(Foreign)', ', 뒤면(Foreign)', ', 식(Foreign)'
                     ]

    
    missing_Josa = [', 이면(Foreign)', ', 이랑(Foreign)', ', 이라서(Foreign)', ', 이라도(Foreign)',
                    ', 이라고(Foreign)', ', 이라(Foreign)', ', 이나(Foreign)', ', 이고(Foreign)',
                    ', 이(Foreign)', ', 의(Foreign)', ', 을(Foreign)', ', 은(Foreign)',
                    ', 으로(Foreign)', ', 엔(Foreign)', ', 에서(Foreign)', ', 에도(Foreign)',
                    ', 에는(Foreign)', ', 에(Foreign)', ', 면(Foreign)', ', 로(Foreign)',
                    ', 는(Foreign)', ', 나(Foreign)', ', 가(Foreign)', ', 라고(Foreign)'
                    ', 이라는(Foreign)', ', 께(Foreign)', ', 를(Foreign)', ', 께(Foreign)',
                    ', 고(Foreign)'
                   ]
    
    
    missing_Verb = [', 하면(Foreign)', ', 하고(Foreign)', ', 주고(Foreign)', ', 되면(Foreign)',
                    ', 된(Foreign)', ', 해서(Foreign)', ', 이며(Foreign)'
                   ]
    
    
    
    missing_Suffix = [', 어치(Foreign)', ', 치(Foreign)', ', 차(Foreign)', ', 째(Foreign)',
                    ', 짜리(Foreign)', ', 씩(Foreign)', ', 생(Foreign)', ', 명(Foreign)',
                    ', 대(Foreign)', ', 달에(Foreign)', ', 달(Foreign)', ', 도에(Foreign)',
                    ', 도(Foreign)', ', 날(Foreign)', ', 급(Foreign)', ', 언(Foreign)',
                    ', 개(Foreign)', '용(Foreign)', '형(Foreign)', '대의(Foreign)',
                    ', 가량(Foreign)', ', 기(Foreign)', ', 부(Foreign)', ', 급의(Foreign)',
                    ', 제(Foreign)', ', 당(Foreign)', ', 개의(Foreign)', ', 권(Foreign)',
                    ', 불(Foreign)', ', 때(Foreign)', ', 짜리가(Foreign)'
                     ]
    
    
    missing_Noun = [', 도착(Foreign)', ', 퇴근(Foreign)', ', 컷(Foreign)', ', 각(Foreign)',
                    ', 걸림(Foreign)', ', 출발(Foreign)', ', 리즈(Foreign)', ', 도서(Foreign)',
                    ', 펀딩(Foreign)', ', 결제(Foreign)', ', 발송(Foreign)', ', 배송(Foreign)',
                    ', 기준(Foreign)', ', 규정(Foreign)', ', 와디즈(Foreign)', ', 무상(Foreign)',
                    ', 리워드(Foreign)', ', 출근(Foreign)', ', 거리(Foreign)'
                   ]
    
    
    missing_Eomi = [', 입니다(Foreign)', ', 임(Foreign)', ', 인디(Foreign)', ', 인데(Foreign)',
                    ', 인가(Foreign)', ', 이지(Foreign)', ', 이요(Foreign)', ', 이여(Foreign)',
                    ', 이야(Foreign)', ', 이래(Foreign)', ', 이라니(Foreign)', ', 이다(Foreign)',
                    ', 이니까(Foreign)', ', 이네(Foreign)', ', 요(Foreign)', ', 야(Foreign)',
                    ', 라(Foreign)', ', 다(Foreign)', ', 네(Foreign)', ', 이얌(Foreign)',
                    ', 이니(Foreign)', ', 이라는데(Foreign)', ', 대에(Foreign)', ', 니까(Foreign)',
                    ', 이넹(Foreign)', ', 이던데(Foreign)', ', 지(Foreign)', ', 에나(Foreign)',
                    ', 함(Foreign)', ', 이었는데(Foreign)', ', 이거든(Foreign)', ', 이었나(Foreign)',
                    ', 이에요(Foreign)'
                   ]
    
    missing_list = [missing_Adverb, missing_Josa, missing_Verb, missing_Suffix, missing_Noun, missing_Eomi]
    fit_list = ['(Adverb)', '(Josa)', '(Verb)', '(Suffix)', '(Noun)', '(Eomi)']
    for i in range(len(missing_list)):
        missing = missing_list[i]
        fit = fit_list[i]
        for n in range(len(missing)):
            if missing[n] in tokenized:
                fixed_pos = missing[n].replace('(Foreign)', fit)
                tokenized = tokenized.replace(missing[n], fixed_pos)
            else:
                pass
    return(tokenized)


def fix_suffix_Noun(tokenized):
    missing = [', 오빠(Suffix)', ', 언니(Suffix)', ', 누나(Suffix)', ', 형(Suffix)',
               ', 엄마(Suffix)', ', 아빠(Suffix)', 
               ', User(Alpha)', ', UserUser(Alpha)', ', UserUserUser(Alpha)']
    
    for i in range(len(missing)):
        if missing[i] in tokenized:
            fixed_pos = missing[i].replace('(Suffix)', '(Noun)')
            tokenized = tokenized.replace(missing[i], fixed_pos)
        elif missing[i] in tokenized:
            fixed_pos = missing[i].replace('(Alpha)', '(Noun)')
            tokenized = tokenized.replace(missing[i], fixed_pos)
            
        else:
            pass
    return(tokenized)


def fix_suffix_Noun2(tokenized):
    missing = [', 님(Suffix)', ', 분들(Suffix)']   
    for i in range(len(missing)):
        if missing[i] in tokenized:
            fixed_pos = missing[i].replace('(Suffix)', '(Noun)')
            tokenized = tokenized.replace(missing[i], fixed_pos)
        else:
            pass
    return(tokenized)
```


```python
# 해석이 모호하거나 파괴된 토큰 결헙 및 형태소 재분류
def restore_pos(tokenized_sentence):
    import re
    import numpy as np
    global re_sentence
    global pos_label
    
    token_list = tokenized_sentence.split(', ')

    for i in range(len(token_list)-1):
        ft_index = i
        bt_index = i+1
        
        f_token = token_list[ft_index]
        b_token = token_list[bt_index]
        
        if '(Verb)' in f_token and '(Eomi)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Verb)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        if '(Adjective)' in f_token and '(Eomi)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Adjective)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if '(VerbPrefix)' in f_token and '(VerbPrefix)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
    
    
        if '(VerbPrefix)' in f_token and '(Verb)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Verb)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        if '(VerbPrefix)' in f_token and '(Adjective)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Adjective)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if '(Verb)' in f_token and '(VerbPrefix)' in b_token :
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
        
        if '하(Suffix)' in f_token and '(Josa)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            f_token = assemble_token + '(Verb)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token 
        
        
        if '(Noun)' in f_token and '(Suffix)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
        if '(Adjective)' in f_token and '(Suffix)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            f_token = assemble_token + '(Adjective)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
        if '(Alpha)' in f_token and '(Suffix)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            f_token = assemble_token + '(Noun)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
             
        
        if '(PreEomi)' in f_token and '(Eomi)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Eomi)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
    
        
        if '(Modifier)' in f_token and '(Modifier)' in b_token :           
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
        if '(Modifier)' in f_token and '(Noun)' in b_token :          
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
      

        
    ws_indexs = np.where(np.array(token_list) == '')[0].tolist()
    w = 0
    for ws in ws_indexs:
        ws -= w
        del token_list[ws]
        w += 1
    
    re_sentence = ', '.join(token_list)

    return(re_sentence)
```


```python
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(fix_foreign)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(fix_suffix_Noun)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(restore_pos)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(restore_pos)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(fix_suffix_Noun2)
```

    100%|████████████████████████████████████████████████████████████████████████████| 3502912/3502912 [03:24<00:00, 17136.46it/s]
    ...생략...

```python
tokenized_df.to_csv('350만_Tokenized.csv(pos 교정)', index = False)
```




<br><br>
## 03. 상대 편향도/상대 빈출도 딕셔너리 구축
### 형태소 재분류한 딕셔너리 생성


```python
#메모리아웃 되는 것을 방지하기 위해 데이터를 일정 크기 이하로 Split
def split_dataframe(dataframe):
    total_length = len(dataframe)
    splited_li = []  # splited_li는 반드시 초기화되어야 합니다.

    if len(dataframe) > 100000:
        split_size = 100000
        num_split = total_length // split_size + 1

        for i in range(num_split):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size
            try:
                splited = dataframe[start_idx:end_idx]
            except:
                splited = dataframe[start_idx:]

            # 작은 그룹을 리스트에 추가
            splited_li.append(splited)
    else:
        splited_li.append(dataframe)
    return splited_li
```


```python
def occur_countor(df):
    total_length = len(df)
    splited_li = split_dataframe(df)

    Occur_dict = {}
    for splited_df in splited_li: 
        merge_list = []
        for index, row in tqdm(splited_df.iterrows(), total=len(splited_df), desc="전체 토큰 병합", mininterval=0.1):
            token_list = row['tokenized'].split(', ')
            for token in token_list:
                merge_list.append(token)

        count_list = Counter(merge_list).most_common()

        sort_arry = []
        for d, c in tqdm(count_list, total=len(count_list), desc="빈도순 정렬", mininterval=0.1):
            for i in range(c):
                sort_arry.append(d)

        unique_dic = Counter(sort_arry)
        unique_token = list(unique_dic.keys())
        unique_freq = list(unique_dic.values())
        
        for key, value in zip(unique_token, unique_freq):  # zip을 사용하여 두 리스트를 동시에 순회
            if key in Occur_dict:
                Occur_dict[key] += value
            else:
                Occur_dict[key] = value

    token_li = Occur_dict.keys()
    freq_li = Occur_dict.values()

    Occur_dic = pd.DataFrame({'Token': token_li, 'Token_freq': freq_li})
    Occur_dic = Occur_dic.sort_values(by=['Token_freq'], ascending=False).reset_index(drop=True) 
    Occur_dic['Total_ratio'] = Occur_dic['Token_freq'].apply(lambda x: x/total_length)
    Occur_dic = Occur_dic.loc[Occur_dic['Total_ratio'] >= 0.0001]
    return Occur_dic
```


```python
Occur_dic = occur_countor(tokenized_df)
Occur_dic
```

    전체 토큰 병합: 100%|██████████████████████████████████████████████████████████████| 100000/100000 [00:04<00:00, 23878.38it/s]
    빈도순 정렬: 100%|████████████████████████████████████████████████████████████████| 212146/212146 [00:00<00:00, 502799.24it/s]
    ...생략...




```python
Occur_dic.to_csv('350만_Occur_dic(pos 교정).csv', index=False)
```


<br><br>
### 토큰의 상대 빈출도 및 상대 편향도 계산


```python
def feature_extractor(Data, Occur_dict , col, Feature_list):
    token_dic = Occur_dict['Token'].to_list()
    
    merged_df = pd.DataFrame()
    merged_df['Token'] = Occur_dict['Token']
    for feature in Feature_list:
        feature_df = Data.loc[Data[f'{col}'] == feature]

        Token_Freq_in_feature_Documents = [0 for i in range(len(token_dic))] # 특성 문서에서 특정 토큰이 출현한 횟수 
        feature_Documents_Freq_by_Token = [0 for i in range(len(token_dic))] # 특정 토큰이 출현한 특성 문서의 수
        
        total_length = len(feature_df)
        splited_li = split_dataframe(feature_df)

        for batch, splited_df in enumerate(splited_li):  
            for index, row in tqdm(splited_df.iterrows(), total=len(splited_df), desc=f"{feature}_{batch+1}/{len(splited_li)}", mininterval=0.1):
                token_list = row['tokenized'].split(', ') # 문서 토크나이징  

                for token in token_list: # 문서에서 특정 토큰이 출현한 수
                    if token in token_dic:
                        idx = token_dic.index(token)
                        Token_Freq_in_feature_Documents[idx] += 1

                    else:
                        pass

                uniqe_token_list = list(set(token_list)) # 토크나이징 된 문장에서 고유한 토큰만 남김, 한 문서에서 동일한 토큰이 여러번 카운팅 되는 것을 방지
                for uniqe_token in uniqe_token_list: # 특정 토큰이 출현한 문서의 수
                    if uniqe_token in token_dic: 
                        idx = token_dic.index(uniqe_token)
                        feature_Documents_Freq_by_Token[idx] += 1
                    else:
                        pass  

        Token_Freq_in_feature_Documents_list = []
        for f in Token_Freq_in_feature_Documents:
            if f == 0:
                f = 0.1
            else:
                pass
            Token_Freq_in_feature_Documents_list.append(f)

        feature_Documents_Freq_by_Token_list = []
        for b in feature_Documents_Freq_by_Token:
            if b == 0:
                b = 0.1
            else:
                pass
            feature_Documents_Freq_by_Token_list.append(b)
        
        feature_df = pd.DataFrame()
        feature_df['Token'] = Occur_dic['Token']
        
        feature_Documents_num = len(feature_df)
        feature_df[f'Freq_{feature}'] = Token_Freq_in_feature_Documents_list #성별에 따른 토큰 출현 빈도
        feature_df[f'Freq_ratio_{feature}'] = np.array(feature_df[f'Freq_{feature}'])/feature_Documents_num #성별에 따른 토큰 출현율
        feature_df[f'Bias_{feature}'] = feature_Documents_Freq_by_Token_list #특정 토큰 출현 시, 작성자 성별이 'sex'인 빈도
        feature_df[f'Bias_ratio_{feature}'] = np.array(feature_df[f'Bias_{feature}'])/feature_Documents_num #특정 토큰 출현 시, 작성자 성별이 'sex'인 비율
    
        merged_df = pd.merge(merged_df, feature_df, how='left', on='Token')
    return merged_df 
```


```python
gender_class = ['남성', '여성']
gender_dic = feature_extractor(df, Occur_dic, 'sex', gender_class)
```

    남성_1/8: 100%|██████████████████████████████████████████████████████████████████████| 100000/100000 [11:33<00:00, 144.14it/s]
    ...생략...




```python
age_class = ['20대 미만', '20대', '30대', '40대', '50대', '60대', '70대 이상'] #
age_dic = feature_extractor(df, Occur_dic, 'age', age_class)
```

    20대 미만_1/2: 100%|█████████████████████████████████████████████████████████████████| 100000/100000 [11:40<00:00, 142.79it/s]
    ...생략...




```python
other_dic = pd.DataFrame()
other_dic['Token'] = age_dic['Token']

other10_Douments_num = len(df.loc[df['age'] != '20대 미만'])
other_dic['Freq_ratio_other10'] = (age_dic['Freq_20대'] + age_dic['Freq_30대'] + age_dic['Freq_40대'] + age_dic['Freq_50대'] + age_dic['Freq_60대'] + age_dic['Freq_70대 이상']) / other10_Douments_num # 
other_dic['Bias_ratio_other10'] = (age_dic['Bias_20대'] + age_dic['Bias_30대'] + age_dic['Bias_40대'] + age_dic['Bias_50대'] + age_dic['Bias_60대'] + age_dic['Bias_70대 이상']) / other10_Douments_num # 

other20_Douments_num = len(df.loc[df['age'] != '20대'])
other_dic['Freq_ratio_other20'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_30대'] + age_dic['Freq_40대'] + age_dic['Freq_50대'] + age_dic['Freq_60대'] + age_dic['Freq_70대 이상']) / other20_Douments_num # 
other_dic['Bias_ratio_other20'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_30대'] + age_dic['Bias_40대'] + age_dic['Bias_50대'] + age_dic['Bias_60대'] + age_dic['Bias_70대 이상']) / other20_Douments_num # 

other30_Douments_num = len(df.loc[df['age'] != '30대'])
other_dic['Freq_ratio_other30'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_20대'] + age_dic['Freq_40대'] + age_dic['Freq_50대'] + age_dic['Freq_60대'] + age_dic['Freq_70대 이상']) / other30_Douments_num # 
other_dic['Bias_ratio_other30'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_20대'] + age_dic['Bias_40대'] + age_dic['Bias_50대'] + age_dic['Bias_60대'] + age_dic['Bias_70대 이상']) / other30_Douments_num # 

other40_Douments_num = len(df.loc[df['age'] != '40대'])
other_dic['Freq_ratio_other40'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_20대'] + age_dic['Freq_30대'] + age_dic['Freq_50대'] + age_dic['Freq_60대'] + age_dic['Freq_70대 이상']) / other40_Douments_num # 
other_dic['Bias_ratio_other40'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_20대'] + age_dic['Bias_30대'] + age_dic['Bias_50대'] + age_dic['Bias_60대'] + age_dic['Bias_70대 이상']) / other40_Douments_num # 

other50_Douments_num = len(df.loc[df['age'] != '50대'])
other_dic['Freq_ratio_other50'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_20대'] + age_dic['Freq_30대'] + age_dic['Freq_40대'] + age_dic['Freq_60대'] + age_dic['Freq_70대 이상']) / other50_Douments_num # 
other_dic['Bias_ratio_other50'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_20대'] + age_dic['Bias_30대'] + age_dic['Bias_40대'] + age_dic['Bias_60대'] + age_dic['Bias_70대 이상']) / other50_Douments_num # 

other60_Douments_num = len(df.loc[df['age'] != '60대'])
other_dic['Freq_ratio_other60'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_20대'] + age_dic['Freq_30대'] + age_dic['Freq_40대'] + age_dic['Freq_50대'] + age_dic['Freq_70대 이상']) / other60_Douments_num
other_dic['Bias_ratio_other60'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_20대'] + age_dic['Bias_30대'] + age_dic['Bias_40대'] + age_dic['Bias_50대'] + age_dic['Bias_70대 이상']) / other60_Douments_num

other70_Douments_num = len(df.loc[df['age'] != '70대 이상'])
other_dic['Freq_ratio_other70'] = (age_dic['Freq_20대 미만'] + age_dic['Freq_20대'] + age_dic['Freq_30대'] + age_dic['Freq_40대'] + age_dic['Freq_50대'] + age_dic['Freq_60대']) / other70_Douments_num
other_dic['Bias_ratio_other70'] = (age_dic['Bias_20대 미만'] + age_dic['Bias_20대'] + age_dic['Bias_30대'] + age_dic['Bias_40대'] + age_dic['Bias_50대'] + age_dic['Bias_60대']) / other70_Douments_num
```



```python
relative_dic = pd.DataFrame()
relative_dic['Token'] = Occur_dic['Token']
relative_dic['Token_freq'] = Occur_dic['Token_freq']
relative_dic['Total_ratio'] = Occur_dic['Total_ratio']

#성별에 대한 
relative_dic['Gender_Freq'] = np.log(gender_dic['Freq_ratio_남성']/gender_dic['Freq_ratio_여성']) # 성별 상대 빈출도
relative_dic['Gender_Bias'] = np.log(gender_dic['Bias_ratio_남성']/gender_dic['Bias_ratio_여성']) # 성별 상대 편향도


#연령에 대한
relative_dic['A10_Freq'] = np.log(age_dic['Freq_ratio_20대 미만']/other_dic['Freq_ratio_other10']) 
relative_dic['A10_Bias'] = np.log(age_dic['Bias_ratio_20대 미만']/other_dic['Bias_ratio_other10']) 

relative_dic['A20_Freq'] = np.log(age_dic['Freq_ratio_20대']/other_dic['Freq_ratio_other20']) 
relative_dic['A20_Bias'] = np.log(age_dic['Bias_ratio_20대']/other_dic['Bias_ratio_other20']) 

relative_dic['A30_Freq'] = np.log(age_dic['Freq_ratio_30대']/other_dic['Freq_ratio_other30']) 
relative_dic['A30_Bias'] = np.log(age_dic['Bias_ratio_30대']/other_dic['Bias_ratio_other30'])

relative_dic['A40_Freq'] = np.log(age_dic['Freq_ratio_40대']/other_dic['Freq_ratio_other40']) 
relative_dic['A40_Bias'] = np.log(age_dic['Bias_ratio_40대']/other_dic['Bias_ratio_other40'])

relative_dic['A50_Freq'] = np.log(age_dic['Freq_ratio_50대']/other_dic['Freq_ratio_other50']) 
relative_dic['A50_Bias'] = np.log(age_dic['Bias_ratio_50대']/other_dic['Bias_ratio_other50'])

relative_dic['A60_Freq'] = np.log(age_dic['Freq_ratio_60대']/other_dic['Freq_ratio_other60']) 
relative_dic['A60_Bias'] = np.log(age_dic['Bias_ratio_60대']/other_dic['Bias_ratio_other60'])

relative_dic['A70_Freq'] = np.log(age_dic['Freq_ratio_70대 이상']/other_dic['Freq_ratio_other70']) 
relative_dic['A70_Bias'] = np.log(age_dic['Bias_ratio_70대 이상']/other_dic['Bias_ratio_other70'])

relative_dic
```




```python
def word_seperator_in_list(token):    
    idx_arry = []
    idx = 0
    for t in str(token):
        if t == '(':
            idx_arry.append(idx)
        else:
            pass
        idx += 1

    s_index = idx_arry[-1]
    w = token[:s_index]
    return(w)

def pos_seperator_in_list(token):    
    idx_arry = []
    idx = 0
    for t in str(token):
        if t == '(':
            idx_arry.append(idx)
        else:
            pass
        idx += 1

    s_index = idx_arry[-1]
    p = token[s_index+1:-1]
    return(p)
```


```python
relative_dic['word'] = relative_dic['Token'].apply(lambda x: ''.join(x.split('(')[:-1]))
relative_dic['pos'] = relative_dic['Token'].apply(lambda x: x.split('(')[-1].replace(')', ''))

relative_dic = relative_dic[['Token', 'word', 'pos', 'Token_freq', 'Total_ratio',
                             'Gender_Freq', 'Gender_Bias', 
                             'A10_Freq', 'A10_Bias', 'A20_Freq', 'A20_Bias', 
                             'A30_Freq', 'A30_Bias', 'A40_Freq', 'A40_Bias',
                             'A50_Freq', 'A50_Bias', 'A60_Freq', 'A60_Bias', 'A70_Freq', 'A70_Bias']] 

relative_dic
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Token</th>
      <th>word</th>
      <th>pos</th>
      <th>Token_freq</th>
      <th>Total_ratio</th>
      <th>Gender_Freq</th>
      <th>Gender_Bias</th>
      <th>A10_Freq</th>
      <th>A10_Bias</th>
      <th>A20_Freq</th>
      <th>...</th>
      <th>A30_Freq</th>
      <th>A30_Bias</th>
      <th>A40_Freq</th>
      <th>A40_Bias</th>
      <th>A50_Freq</th>
      <th>A50_Bias</th>
      <th>A60_Freq</th>
      <th>A60_Bias</th>
      <th>A70_Freq</th>
      <th>A70_Bias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>?(Punctuation)</td>
      <td>?</td>
      <td>Punctuation</td>
      <td>1865214</td>
      <td>0.532475</td>
      <td>-1.092652</td>
      <td>-1.125282</td>
      <td>1.582501</td>
      <td>1.639460</td>
      <td>4.709142</td>
      <td>...</td>
      <td>3.807751</td>
      <td>3.728915</td>
      <td>1.845276</td>
      <td>1.620177</td>
      <td>1.337092</td>
      <td>1.128453</td>
      <td>-0.383700</td>
      <td>-0.437217</td>
      <td>-2.976230</td>
      <td>-2.885823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ㅋㅋㅋㅋㅋ(KoreanParticle)</td>
      <td>ㅋㅋㅋㅋㅋ</td>
      <td>KoreanParticle</td>
      <td>1526323</td>
      <td>0.435730</td>
      <td>-1.762353</td>
      <td>-1.680824</td>
      <td>1.680877</td>
      <td>1.665422</td>
      <td>4.995337</td>
      <td>...</td>
      <td>3.763199</td>
      <td>3.676334</td>
      <td>-0.645103</td>
      <td>-0.455351</td>
      <td>-1.662415</td>
      <td>-1.496536</td>
      <td>-3.916414</td>
      <td>-3.791524</td>
      <td>-11.291175</td>
      <td>-10.853299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>에(Josa)</td>
      <td>에</td>
      <td>Josa</td>
      <td>1376727</td>
      <td>0.393024</td>
      <td>-1.354588</td>
      <td>-1.337686</td>
      <td>1.516475</td>
      <td>1.556426</td>
      <td>4.815396</td>
      <td>...</td>
      <td>3.781199</td>
      <td>3.756026</td>
      <td>1.484062</td>
      <td>1.430571</td>
      <td>0.922323</td>
      <td>0.894123</td>
      <td>-0.292875</td>
      <td>-0.305520</td>
      <td>-3.032982</td>
      <td>-3.025299</td>
    </tr>
    <tr>
      <th>3</th>
      <td>가(Josa)</td>
      <td>가</td>
      <td>Josa</td>
      <td>1144262</td>
      <td>0.326660</td>
      <td>-1.330090</td>
      <td>-1.316493</td>
      <td>1.611810</td>
      <td>1.626960</td>
      <td>4.836498</td>
      <td>...</td>
      <td>3.745612</td>
      <td>3.730188</td>
      <td>1.468008</td>
      <td>1.432860</td>
      <td>0.856439</td>
      <td>0.860018</td>
      <td>-0.144169</td>
      <td>-0.195058</td>
      <td>-2.928766</td>
      <td>-2.935806</td>
    </tr>
    <tr>
      <th>4</th>
      <td>이(Josa)</td>
      <td>이</td>
      <td>Josa</td>
      <td>919870</td>
      <td>0.262602</td>
      <td>-1.318207</td>
      <td>-1.311745</td>
      <td>1.399605</td>
      <td>1.440919</td>
      <td>4.757490</td>
      <td>...</td>
      <td>3.840997</td>
      <td>3.813802</td>
      <td>1.523747</td>
      <td>1.475925</td>
      <td>1.019233</td>
      <td>0.979362</td>
      <td>0.074839</td>
      <td>-0.029539</td>
      <td>-2.691890</td>
      <td>-2.770597</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18356</th>
      <td>암데(Noun)</td>
      <td>암데</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
      <td>-1.481605</td>
      <td>-1.486564</td>
      <td>1.325666</td>
      <td>1.337359</td>
      <td>4.437628</td>
      <td>...</td>
      <td>4.217711</td>
      <td>4.220826</td>
      <td>1.701743</td>
      <td>1.713539</td>
      <td>0.075802</td>
      <td>0.087326</td>
      <td>0.492802</td>
      <td>0.095993</td>
      <td>-2.913589</td>
      <td>-2.902128</td>
    </tr>
    <tr>
      <th>18357</th>
      <td>라드(Noun)</td>
      <td>라드</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
      <td>-1.094817</td>
      <td>-1.090738</td>
      <td>0.757370</td>
      <td>0.795543</td>
      <td>4.912486</td>
      <td>...</td>
      <td>3.725778</td>
      <td>3.774007</td>
      <td>1.336311</td>
      <td>1.374823</td>
      <td>0.774695</td>
      <td>0.812868</td>
      <td>0.783361</td>
      <td>0.821534</td>
      <td>-2.913589</td>
      <td>-2.875849</td>
    </tr>
    <tr>
      <th>18358</th>
      <td>액상(Noun)</td>
      <td>액상</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
      <td>-0.445995</td>
      <td>-0.421465</td>
      <td>0.466524</td>
      <td>0.173400</td>
      <td>5.630803</td>
      <td>...</td>
      <td>3.085743</td>
      <td>3.107728</td>
      <td>0.767727</td>
      <td>0.184045</td>
      <td>0.075516</td>
      <td>0.190725</td>
      <td>-2.916976</td>
      <td>-2.802428</td>
      <td>-2.913874</td>
      <td>-2.799325</td>
    </tr>
    <tr>
      <th>18359</th>
      <td>스티(Noun)</td>
      <td>스티</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
      <td>-1.704748</td>
      <td>-1.638637</td>
      <td>0.983399</td>
      <td>1.039861</td>
      <td>4.927054</td>
      <td>...</td>
      <td>3.824710</td>
      <td>3.831202</td>
      <td>0.477456</td>
      <td>0.533584</td>
      <td>0.075802</td>
      <td>0.131765</td>
      <td>0.084469</td>
      <td>0.140431</td>
      <td>-2.913589</td>
      <td>-2.857938</td>
    </tr>
    <tr>
      <th>18380</th>
      <td>다른팀(Noun)</td>
      <td>다른팀</td>
      <td>Noun</td>
      <td>351</td>
      <td>0.000100</td>
      <td>-1.500599</td>
      <td>-1.477266</td>
      <td>0.466524</td>
      <td>0.501595</td>
      <td>5.463502</td>
      <td>...</td>
      <td>3.345351</td>
      <td>3.364846</td>
      <td>0.068836</td>
      <td>0.103804</td>
      <td>-0.620491</td>
      <td>-0.585624</td>
      <td>-2.916976</td>
      <td>-2.882200</td>
      <td>-2.913874</td>
      <td>-2.879098</td>
    </tr>
  </tbody>
</table>
<p>18381 rows × 21 columns</p>






```python
relative_dic.to_csv('350만_feature_dic.csv', index=False)
```


