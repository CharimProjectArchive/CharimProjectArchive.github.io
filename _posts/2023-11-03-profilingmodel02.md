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
 
 
## 변수 개발 요약
**변수 개발의 필요성**
- 줄임말, 고유 표현, 자음 또는 모음으로만 구성된 불완전 표현, 문장기호, 특수기호, 이모지 등의 표현은 **기존의 NLP 전처리 및 언어모델의 Encoding 과정에서 제거됨**. 하지만, 이와 같은 표현들이 **발화자의 성별/연령에 대한 언어적 특징으로 보여짐으로** 상관관계 파악 및 모델개발 시 반영을 위한 feauture 개발 필요
- 모든 표현을 encoding하는 방식은 비효율적일뿐만 아니라 사전 사전 학습된 모델에 적용할 수 없으므로 **언어적의 특징을 계량화한 변수**가 요구됨
<br>

 
### 1. 표현의 길이와 철자의 다양성을 바탕으로 한 복잡도 계산
**Complexity of Specific Expression(C, 특수표현 복잡도)** 
- 표현 t가 복잡한 정도를 설명함
- 전처리에서 대부분 소실되는 자음 또는 모음으로 구성된 표현, 문장기호, 특수기호, 이모지로 구성된 표현에 대한 복잡도를 계산
- 자음 또는 모음으로 구성된 표현을 한 분류로 하고, 문장기호/특수기호/이모지를 한 분류로 구분하여 계산(형태소 분류 및 정규식 처리에 용이)
> $C_i= ln[ 1/Π(U_i / L_i) * L_i]$<br>
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
> $RBG_i = ln[ n(t_i∈s_{male})/n(s_{male}) ÷ n(t_i∈s_{female})/ n(s_{female}) ]$<br>
> $t_i$ : 문서 내 i번 째 표현
> $s_{male}$ : 작성자의 성별이 남성(m)인 문장
> $s_{female}$ : 작성자의 성별이 여성(f)인 문장
> <br><br>
> 작성자의 성별이 남성인 문장 $s_{male}$ 에서 표현 $t_i$ 가 등장한 비율 ÷ 여성인 문장 $s_{female}$ 에서 표현 $t_i$ 가 등장한 비율, 0~1 사이의 Skewed한 값을 가짐으로 log를 취해 정규화<br>
  ⇒ $t_i$ 가 등장했을 때 작성자의 성별이 남자 $m$ 또는 여자 $f$ 인 정도

<br><br>
 b. **Relative Bias of Age**(RBG, 연령에 대한 상대 편향도)<br>
  - 표현 $t_i$ 가 등장한 문장 $s$ 의 작성자 연령이 특정 연령대 $age$ 인 정도를 설명함<br>
  - 연령대는 20대 미만/20대/30대/40대/50대 이상 5 class로 분류
> $RBAi = ln[ n(t_i∈s_{age})/n(s_{age}) ÷ n(t_i∈s_{other})/n(s_{other}) ]$<br>
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
>  $RFG_i = ln[ n(t_i | s_{male}) / n(s_{male}) ÷ n(t_i | s_{female}) / n(s_{female}) ]$<br>
>  $RFAi = ln[ n(t_i | s_{age}) / n(s_{age}) ÷ n(t_i | s_{other}) / n(s_{other}) ]$


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

df = pd.read_csv("카톡대화_Dataset(raw_중복된 메세지 제거).csv")
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
df['spell_num'],df['spell_mean'], df['spell_std'], df['spell_max'], df['spell_min'] = zip(*df['contents'].progress_apply(lambda x: spell_complexity(str(x))))
df['symbol_num'], df['symbol_mean'],  df['symbol_std'], df['symbol_max'], df['symbol_min'] = zip(*df['contents'].progress_apply(lambda x: symbol_complexity(str(x))))

df = df.drop(columns=['topic', 'resident'])
df = df.fillna(0)
df
```

    100%|████████████████████████████████████████████████████████████████████████████| 3563533/3563533 [04:53<00:00, 12152.45it/s]
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
df.to_csv('카톡대화_Dataset(raw_표현 복잡도 계산).csv', index=False)
```



<br><br>
## 02. 자연어 전처리
- 3번 이상 반복되는 동일 음절은 3음절로 표제화 *ex. ㅇㅇㅇㅇㅇ ⇒ ㅇㅇㅇ
- 문장기호, 특수기호, 이모지, 특수폰트 표현 표제화 *ex. ❤🧡💛 ⇒ ♥️♥️♥️ / ヲ𐨛𐌅⫬ ⇒ ㅋㅋㅋㅋ
- 연속된 띄어쓰기(공백) 한 번으로 처리
- 5어절 미만 말뭉치 데이터 제거


  
```python
processed_df = df[['sex', 'age', 'contents', 'length']].copy()
```



```python
import re

raw = processed_df.shape[0]
print('- raw data: {:,}'.format(raw))

#결측값 제거
print('- null data: {:,}'.format(processed_df['contents'].isnull().sum()))
processed_df = processed_df.dropna() 
deleted_null = processed_df.shape[0]

#중복 제거
processed_df = processed_df.drop_duplicates() 
deleted_dup = processed_df.shape[0]
print('- duplicated data: {:,}'.format(deleted_null - deleted_dup))
```

    - raw data: 3,564,042
    - null data: 0
    - duplicated data: 0


<br><br>
### 자모 표현, 숫자, 기호가 결합된 표현 유형별 분리(띄어쓰기)

```python
def split_re_types(sentence):
    KoreanParticle_pattern = r'[ㄱ-ㅎㅏ-ㅣ]+'
    Symbol_pattern = r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]+'
    Number_pattern = r'(\d+)'
    
    sentence = re.sub(KoreanParticle_pattern, r' \g<0> ', sentence)
    sentence = re.sub(Symbol_pattern, r' \g<0> ', sentence)
    sentence = re.sub(Number_pattern, r' \g<0> ', sentence)
    
    return sentence
```
    
<br><br>
### 주요 이모지 및 특수표현 표제화


```python
def specific_lemmatization(sentence):
    heart_emoji = ['♡', '♥', '❤', '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '💕', '❣️'] #
    star_emoji = ['☆', '★', '⭐', '🌟']
    kkk = ['𐨛', '𐌅', '⫬', 'ヲ', '刁', '㉪', 'ｦ', 'ㅋ꙼̈', 'ㅋ̑̈', 'ㅋ̆̎', 'ㅋ̐̈', 'ㅋ̊̈', 'ㅋ̄̈', 'ㅋ̆̈', 'ㅋ̊̈', 'ㅋ̐̈', 'ㅋ̆̎']
    Period = ['ㆍ', 'ᆞ', 'ㆍ', '•', 'ᆢ']
    quote = ['”', '‘', '“']
    ect = [' ', 'ㅤ']
    
    for h in heart_emoji:
        sentence = sentence.replace(h, '♥')
    for s in star_emoji:
        sentence = sentence.replace(s, '★')
    for k in kkk:
        sentence = sentence.replace(k, 'ㅋ')
    for p in Period:
        sentence = sentence.replace(p, '·')
    for q in quote:
        sentence = sentence.replace(q, '\'')
    for e in ect:
        sentence = sentence.replace(e, ' ')   
    return sentence

text_sentence = '❤🧡💛테스트💚💙💜☆입니다★ᆢ⭐ ヲ𐨛𐌅⫬ㅋ̄̈ㅋ꙼̈ㅋ̆̎ㅋ̐̈ㅋ̊̈ㅋ̄̈ㅋ꙼̈ㅋ̆̎ㅋ̐̈ㅋ̊̈'
text_sentence = specific_lemmatization(text_sentence)
text_sentence
```




    '♥♥♥테스트♥♥♥★입니다★.★ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'




    
<br><br>
### 반복되는 동일 철자 표제화 처리


```python
def duplicated_character_reduction(sentence):
    words = sentence.split()

    for word in words:
        pattern = re.compile(r'([^0-9])\1{4,}')
        re_word = pattern.sub(r'\1' * 5, word)
        sentence = sentence.replace(word, re_word)
    return sentence

text_sentence = '      ^^^^^^^ !!!!!!! 안녕안녕 헤헤헤헤헤헤헤헤 ㅋㅋㅋㅋ ㅎㅎㅎㅎㅎㅎㅎㅎ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'
duplicated_character_reduction(text_sentence)
```




    '      ^^^^^ !!!!! 안녕안녕 헤헤헤헤헤 ㅋㅋㅋㅋ ㅎㅎㅎㅎㅎ ㅋㅋㅋㅋㅋ'




```python
processed_df['contents'] = processed_df['contents'].progress_apply(split_re_types)
processed_df['contents'] = processed_df['contents'].progress_apply(specific_lemmatization)
processed_df['contents'] = processed_df['contents'].progress_apply(duplicated_character_reduction)
```

    100%|████████████████████████████████████████████████████████████████████████████| 3563533/3563533 [02:02<00:00, 29125.25it/s]
    ...생략...
    
<br><br>
### 연속된 띄어쓰기 처리/5어절 미만 문장 제거


```python
#연속되 띄어쓰기 처리
processed_df['contents'] = processed_df['contents'].apply(lambda x : re.sub(r'\s+', ' ', x))  #연속된 띄어쓰기 ' '로 변경
mask = processed_df['contents'].isin([' '])
processed_df = processed_df[~mask].reset_index(drop = True) 
deleted_white = processed_df.shape[0]
white = deleted_dup - deleted_white
print("- white space data: {:,}({}%)".format(white, round(white/deleted_dup*100, 2)))

#어절 카운팅
processed_df['word_bunch'] = processed_df['contents'].apply(lambda x: len(x.split(' ')))


#5어절 미만 문장 제거
cutoff = processed_df.loc[processed_df['word_bunch'] < 5].shape[0] 
processed_df = processed_df.loc[processed_df['word_bunch'] >= 5] 
deleted_cutoff = processed_df.shape[0]
print("- cutoff data: {:,}({}%)".format(cutoff, round(cutoff/deleted_white*100, 2)))

print("- total processed data: {:,}".format(deleted_cutoff))
```

    - white space data: 0(0.0%)
    - cutoff data: 9,073(0.25%)
    - total processed data: 3,554,460


```python
processed_df.to_csv('카톡대화_Dataset(전처리).csv', index=False)
```


<br><br>
### 토크나이즈
- konlpy 라이브러리의 Okt를 사용하여 형태소 분리
- 띄어쓰기를 유지하기 위해 '_(_)'로 대신
- 동일 철자이나 다른 형태소를 갖는 토큰을 파악하기 위해 '토큰(형태소)' 형식으로 토크나이즈

```python
import konlpy

#메모리아웃 되는 것을 방지하기 위해 데이터를 일정 크기 이하로 Split
def split_dataframe(dataframe, size=1000):
    total_length = len(dataframe)
    splited_li = []  # splited_li는 반드시 초기화되어야 합니다.

    if len(dataframe) > size:
        split_size = size
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


def pos_tokenizer(sentence):    #POS 기준 토크나이제이션
    tokens = sentence.split()
    with_space = ' SP '.join(tokens)
    
    token_li = []
    okt = konlpy.tag.Okt() 
    pos_list = okt.pos(str(with_space)) #토큰/형태소 튜플처리
    for t, m in pos_list:
        token = '{}({})'.format(t, m)
        token_li.append(token)
    
    token_count = len(token_li)
    tokenized_sentence = ' '.join(token_li)
    tokenized_sentence = tokenized_sentence.replace('SP(Alpha)', '_(_)')
    return tokenized_sentence, token_count
```




```python
import gc

df_li = split_dataframe(df, size=1000) # 데이터가 커서 분리하여 처리
tokenized_df = pd.DataFrame()
i = 0
for d in tqdm(df_li, total=len(df_li), desc='토크나이징'):
    data = d.copy()
    data['tokenized'], data['token_count'] = zip(*data['contents'].apply(lambda x: pos_tokenizer(x)))
    tokenized_df = pd.concat([tokenized_df, data], ignore_index=True)
    
    i += 1
    if i%500 == 0:
        gc.collect()
    
tokenized_df
```

    토크나이징: 100%|███████████████████████████████████████████████████████████████████████| 3555/3555 [3:20:39<00:00,  3.44s/it]





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>age</th>
      <th>contents</th>
      <th>length</th>
      <th>word_bunch</th>
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
      <td>127</td>
      <td>16</td>
      <td>나(Noun) 지금(Noun) 밥(Noun) 머거(Verb) 2시간(Number) ...</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>남성</td>
      <td>20대</td>
      <td>헐 ㅠㅠ 언넝호텔들가ㅠㅠ 엄청피건할첸데 나는인낫러요 나 두시출근이다ㅎㅎㅎㅎ 퀵으로한...</td>
      <td>130</td>
      <td>18</td>
      <td>헐(Verb) ㅠㅠ(KoreanParticle) 언(Modifier) 넝(Noun)...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>여성</td>
      <td>20대</td>
      <td>학생이면좋구! 왜혼자다니냐고오..... 와 내친군학교나감 ㅋㅋㅋㅋㅋ 그르네 막졸업한...</td>
      <td>56</td>
      <td>7</td>
      <td>학생(Noun) 이(Suffix) 면(Josa) 좋구(Adjective) !(Pun...</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남성</td>
      <td>20대</td>
      <td>훔 학생 없는데...주변에... 아니 복학하고 학교를 못가는데 어케 친구가있냐.. ...</td>
      <td>74</td>
      <td>15</td>
      <td>훔(Noun) 학생(Noun) 없는데(Adjective) ...(Punctuatio...</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>여성</td>
      <td>30대</td>
      <td>참나 내가뭐얼마나그랬다고 웃기는사람이야지짜 너무화난당.. 근데오빠는말을또 잘해서 내...</td>
      <td>146</td>
      <td>19</td>
      <td>참나(Noun) 내(Noun) 가(Josa) 뭐(Noun) 얼마나(Noun) 그랬다...</td>
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
    </tr>
  </tbody>
</table>
<p>3554460 rows × 7 columns</p>




```python
tokenized_df.to_csv('카톡대화_Tokenized(pos 비교정).csv', index = False)
```


<br><br>
### 토큰 딕셔너리 구축 및 스크리닝
**딕셔너리 구축**
- 빈도 0.0001 이상의 토큰만 사용<br>
[토큰 딕셔너리(구글 스프레드)](https://docs.google.com/spreadsheets/d/1wKv4hAfJD_ToORv1Q7QGWsCjYUQTZKHRzZMNCxVPvZ4/edit?usp=sharing)<br>
https://docs.google.com/spreadsheets/d/1wKv4hAfJD_ToORv1Q7QGWsCjYUQTZKHRzZMNCxVPvZ4/edit?usp=sharing

**스크리닝 결과**
- 띄어쓰기 기준으로 토크나이즈된 딕셔너리, 형태소 기준으로 토크나이즈된 딕셔너리, 원 문장, 토크나이즈 문장을 비교

  
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
from collections import Counter

# 띄어쓰기 기준의 토크나이저
def space_token_countor(df):
    total_length = len(df)
    splited_li = split_dataframe(df, size=1000)

    space_dict = {}
    for splited_df in tqdm(splited_li, total=len(splited_li), desc="어절 사전 생성"): 
        merge_list = []
        for index, row in splited_df.iterrows():
            sentence = row['contents'].split()
            for token in sentence:
                merge_list.append(token)

        unique_dic = Counter(merge_list)
        for key, value in unique_dic.items():
            if key in space_dict:
                space_dict[key] += value
            else:
                space_dict[key] = value

    token_li = space_dict.keys()
    freq_li = space_dict.values()

    space_dic = pd.DataFrame({'Token': token_li, 'Token_freq': freq_li})
    space_dic = space_dic.sort_values(by=['Token_freq'], ascending=False).reset_index(drop=True) 
    space_dic['Total_ratio'] = space_dic['Token_freq'].apply(lambda x: x/total_length)
    space_dic = space_dic.loc[space_dic['Token_freq'] >= 300]
    return space_dic


# 형태소 기준의 토크나이저
def pos_token_countor(df):
    total_length = len(df)
    splited_li = split_dataframe(df, size=1000)

    occur_dict = {}
    for splited_df in tqdm(splited_li, total=len(splited_li), desc="형태소 사전 생성"): 
        merge_list = []
        for index, row in splited_df.iterrows():
            token_list = row['tokenized'].split()
            for token in token_list:
                merge_list.append(token)

        unique_dic = Counter(merge_list)
        
        for key, value in unique_dic.items():
            if key in occur_dict:
                occur_dict[key] += value
            else:
                occur_dict[key] = value

    token_li = occur_dict.keys()
    freq_li = occur_dict.values()

    occur_dic = pd.DataFrame({'Token': token_li, 'Token_freq': freq_li})
    occur_dic = occur_dic.sort_values(by=['Token_freq'], ascending=False).reset_index(drop=True) 
    occur_dic['Total_ratio'] = occur_dic['Token_freq'].apply(lambda x: x/total_length)
    occur_dic = occur_dic.loc[occur_dic['Total_ratio'] >= 0.0001]
    return occur_dic

def word_pos_split(token):
    index_li = []
    for i, char in enumerate(token):
        if char == '(':
            index_li.append(i)
    return token[:index_li[-1]], token[index_li[-1]+1:-1]
```

```python
space_dic = space_token_countor(df)
space_dic
```


```python
pos_dic = pos_token_countor(tokenized_df)
pos_dic['word'], pos_dic['pos'] = zip(*pos_dic['Token'].apply(lambda x: word_pos_split(x)))
pos_dic = pos_dic[['Token', 'word', 'pos', 'Token_freq', 'Total_ratio']]
pos_dic
```

    딕셔너리 구성:  100%|████████████████████████████████████████████████████████████████████████████| 3503/3503 [03:24<00:00, 17136.46it/s]

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
      <td>0.532477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ㅋㅋㅋㅋㅋ(KoreanParticle)</td>
      <td>ㅋㅋㅋㅋㅋ</td>
      <td>KoreanParticle</td>
      <td>1454297</td>
      <td>0.415169</td>
    </tr>
    <tr>
      <th>2</th>
      <td>에(Josa)</td>
      <td>에</td>
      <td>Josa</td>
      <td>1307936</td>
      <td>0.373386</td>
    </tr>
    <tr>
      <th>3</th>
      <td>가(Josa)</td>
      <td>가</td>
      <td>Josa</td>
      <td>1142959</td>
      <td>0.326289</td>
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
  </tbody>
</table>
<p>18194 rows × 5 columns</p>



```python
pos_dic.to_csv('카톡대화_pos_dic.csv', index=False)
```




<br><br>
### 주요 오분류  형태소 재분류


```python
# 오분류되는 주요 토큰 형태소 재분류
def fix_foreign(tokenized):
    missing_Adverb = ['후에(Foreign)', '후(Foreign)', '초에(Foreign)', '쯤에(Foreign)',
                      '쯤(Foreign)', '정도에(Foreign)', '정도(Foreign)', '전에는(Foreign)',
                      '전에(Foreign)', '전(Foreign)', '이상(Foreign)', '안에(Foreign)',
                      '부터는(Foreign)', '부터(Foreign)', '반쯤(Foreign)', '반에(Foreign)',
                      '반부터(Foreign)', '반까지(Foreign)', '반(Foreign)', '밖에(Foreign)',
                      '말에(Foreign)', '말(Foreign)', '만에(Foreign)', '만(Foreign)',
                      '마다(Foreign)', '뒤에(Foreign)', '뒤(Foreign)', '동안(Foreign)',
                    
                      '넘어서(Foreign)', '넘게(Foreign)', '남음(Foreign)', '꺼(Foreign)',
                      '까진데(Foreign)', '까진(Foreign)', '까지야(Foreign)', '까지만(Foreign)',
                      '까지는(Foreign)', '까지(Foreign)', '간(Foreign)', '이후(Foreign)',
                      '이후에(Foreign)', '내로(Foreign)', '경(Foreign)', '말까지(Foreign)',
                      '전까지(Foreign)', '중에(Foreign)', '즘(Foreign)', '내내(Foreign)',
                      '정도는(Foreign)', '초(Foreign)', '얼마(Foreign)', '정도면(Foreign)',
                      '이내(Foreign)', '내(Foreign)', '간의(Foreign)', '간은(Foreign)',
                      '약(Foreign)', '보다(Foreign)', '전엔(Foreign)', '까지니까(Foreign)',
                      '정도만(Foreign)', '사이에(Foreign)', '뒤면(Foreign)', '식(Foreign)'
                     ]

    
    missing_Josa = ['이면(Foreign)', '이랑(Foreign)', '이라서(Foreign)', '이라도(Foreign)',
                    '이라고(Foreign)', '이라(Foreign)', '이나(Foreign)', '이고(Foreign)',
                    '이(Foreign)', '의(Foreign)', '을(Foreign)', '은(Foreign)',
                    '으로(Foreign)', '엔(Foreign)', '에서(Foreign)', '에도(Foreign)',
                    '에는(Foreign)', '에(Foreign)', '면(Foreign)', '로(Foreign)',
                    '는(Foreign)', '나(Foreign)', '가(Foreign)', '라고(Foreign)'
                    '이라는(Foreign)', '께(Foreign)', '를(Foreign)', '께(Foreign)',
                    '고(Foreign)'
                   ]
    
    
    missing_Verb = ['하면(Foreign)', '하고(Foreign)', '주고(Foreign)', '되면(Foreign)',
                    '된(Foreign)', '해서(Foreign)', '이며(Foreign)'
                   ]
    
    
    
    missing_Suffix = ['어치(Foreign)', '치(Foreign)', '차(Foreign)', '째(Foreign)',
                      '짜리(Foreign)', '씩(Foreign)', '생(Foreign)', '명(Foreign)',
                      '대(Foreign)', '달에(Foreign)', '달(Foreign)', '도에(Foreign)',
                      '도(Foreign)', '날(Foreign)', '급(Foreign)', '언(Foreign)',
                      '개(Foreign)', '용(Foreign)', '형(Foreign)', '대의(Foreign)',
                      '가량(Foreign)', '기(Foreign)', '부(Foreign)', '급의(Foreign)',
                      '제(Foreign)', '당(Foreign)', '개의(Foreign)', '권(Foreign)',
                      '불(Foreign)', '때(Foreign)', '짜리가(Foreign)'
                     ]
    
    
    missing_Noun = ['도착(Foreign)', '퇴근(Foreign)', '컷(Foreign)', '각(Foreign)',
                    '걸림(Foreign)', '출발(Foreign)', '리즈(Foreign)', '도서(Foreign)',
                    '펀딩(Foreign)', '결제(Foreign)', '발송(Foreign)', '배송(Foreign)',
                    '기준(Foreign)', '규정(Foreign)', '와디즈(Foreign)', '무상(Foreign)',
                    '리워드(Foreign)', '출근(Foreign)', '거리(Foreign)'
                   ]
    
    
    missing_Eomi = ['입니다(Foreign)', '임(Foreign)', '인디(Foreign)', '인데(Foreign)',
                    '인가(Foreign)', '이지(Foreign)', '이요(Foreign)', ',여(Foreign)',
                    '이야(Foreign)', '이래(Foreign)', '이라니(Foreign)', ', 이다(Foreign)',
                    '이니까(Foreign)', '이네(Foreign)', '요(Foreign)', '야(Foreign)',
                    '라(Foreign)', '다(Foreign)', '네(Foreign)', ', 이얌(Foreign)',
                    '이니(Foreign)', '이라는데(Foreign)', '대에(Foreign)', '니까(Foreign)',
                    '이넹(Foreign)', '이던데(Foreign)', '지(Foreign)', '에나(Foreign)',
                    '함(Foreign)', '이었는데(Foreign)', '이거든(Foreign)', '이었나(Foreign)',
                    '이에요(Foreign)'
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
    missing = ['오빠(Suffix)', '언니(Suffix)', '누나(Suffix)', '형(Suffix)',
               '엄마(Suffix)', '아빠(Suffix)', '님(Suffix)', '분들(Suffix)',
               'User(Alpha)', 'UserUser(Alpha)', 'UserUserUser(Alpha)']
    
    for i in range(len(missing)):
        if missing[i] in tokenized:
            fixed_pos = missing[i].replace('(Suffix)', '(Noun)').replace('(Alpha)', '(Noun)')
            tokenized = tokenized.replace(missing[i], fixed_pos)

    return(tokenized)
```


```python
# 해석이 모호하거나 파괴된 토큰 결헙 및 형태소 재분류
single_words = ['할인가',  '할인가격', '할인금', '할인금액',
                '펀딩가', '펀딩가격', '펀딩금', '펀딩금액',
                '예정가', '예정가격',
                '정상가', '정상가격', 
                '결제일',
                '알림신청',
                '얼리버드',
                '리워드', 
                '사은품',
                '구성품',
                '수령일',
                '배송일',
                '종료일',
                '택배사',
                '택배발송',
                '보증기간',
                'C타입',
                '새소식',
                '고객샌터', '고객센터',
                '후원금', '후원금액',
                '모델명',
                '크라우드',
                '특허증',
                '사용법', '사용방법',
                '제조국',
                '생산지',
                '접수처',
                '콜드브루', '콜드부르',
                '키보드',
                '받침대',
                '맞춤형',
                '가열식',
                '일체형',
                '끝판왕',
                '전문가용',
                '숙련자', '숙련자용',
                '명암비',
                '풀패키지',
                '와디즈',
                '아답터'
                '서포터즈'
                '안내', '안내문'
                '다들',
                '도착',
                '그쵸',
                '충동구매'
               ]

destroyed_words = {'할인(Noun)' : ['가', '금'],
                   '펀딩(Noun)' : ['가', '금'],
                   '예정(Noun)' : ['가'],
                   '정상(Noun)' : ['가'],
                   '결제(Noun)' : ['일'],
                   '알림(Noun)' : ['신'],
                   '얼리(Verb)' : ['버'],
                   '리(Noun)' : ['워'], 
                   '사은(Noun)': ['품'],
                   '구(Modifier)' : ['성'],
                   '수령(Noun)' : ['일'],
                   '배송(Noun)' : ['일'],
                   '종료(Noun)' : ['일'],
                   '택배(Noun)' : ['사', '발'],
                   '보증(Noun)' : ['기간'],
                   'C(Alpha)' : ['타'],
                   '새(Modifier)' : ['소'],
                   '고객(Noun)' : ['샌', '센'],
                   '후(Noun)' : ['원'],
                   '후원(Noun)' : ['금'],
                   '모델(Noun)' : ['명'],
                   '크라(Verb)' : ['우'],       
                   '특허(Noun)' : ['증'],
                   '사용(Noun)' : ['법', '방'],
                   '사(Modifier)' : ['용'],
                   '제(Modifier)' : ['조'],
                   '생산(Noun)' : ['지'],
                   '접수(Noun)' : ['처'],
                   '콜드(Noun)' : ['브', '부'],
                   '키(Noun)' : ['보'],
                   '받침(Noun)' : ['대'],
                   '맞춤(Noun)' : ['형'],
                   '가열(Noun)' : ['식'],
                   '일체(Noun)' : ['형'],
                   '끝판(Noun)' : ['왕'],
                   '전문(Noun)' : ['가'],
                   '숙련(Noun)' : ['자'],
                   '명암(Noun)' : ['비'],
                   '풀(Noun)' : ['패'],
                   '와디(Noun)' : ['즈'],
                   '아(Exclamation)' : ['답'],
                   '서포터(Noun)' :['즈'],
                   '안(VerbPrefix)' : ['내'],
                   '안내(VerbPrefix)' : ['문'],
                   '다(Adverb)' : ['들'],
                   '도(Suffix)' : ['착'],
                   '그(Noun)' : ['쵸'],
                   '충동(Noun)' : ['구'],
                   '들(Suffix)' : ['가']
                  }



def restore_word(tokenized_sentence):
    import re
    import numpy as np
    global re_sentence
    global pos_label
    
    token_list = tokenized_sentence.split()
    
    destroyed_keys = destroyed_words.keys()
    for dest_front in destroyed_keys:
        if dest_front in token_list:
            fw_indexs = np.where(np.array(token_list) == dest_front)[0].tolist()
            fron_word = dest_front
            
            for fw_index in fw_indexs:
                bw_index = int(fw_index + 1)
                
                if bw_index < len(token_list):
                    back_word = token_list[bw_index]
                    
                    dest_back_list = destroyed_words[dest_front]
                    for dest_back in dest_back_list:
                        if dest_back == back_word[0]:
                            
                            assemble_word = re.sub(pattern = r'\([^)]*\)', repl='', string = str(fron_word + back_word))

                            for s_word in single_words:
                                if s_word in assemble_word and s_word == assemble_word:
                                    re_fw = assemble_word + '(Noun)'

                                    token_list[fw_index] = re_fw
                                    token_list[bw_index] = ''

                                elif s_word in assemble_word and s_word != assemble_word:
                                    sw_lenght = len(s_word)
                                    re_fw = assemble_word[:sw_lenght] + '(Noun)'
                                    re_bw = assemble_word[sw_lenght:] + '(Josa)'

                                    token_list[fw_index] = re_fw
                                    token_list[bw_index] = re_bw
                                    
    ws_indexs = np.where(np.array(token_list) == '')[0].tolist()
    w = 0
    for ws in ws_indexs:
        ws -= w
        del token_list[ws]
        w += 1
    
    re_sentence = ' '.join(token_list)

    return(re_sentence)
```

```python
def refine_pos1(tokenized_sentence):
    token_list = tokenized_sentence.split()
    
    for i in range(len(token_list)-1):
        ft_index = i
        bt_index = i+1
        
        f_token = token_list[ft_index]
        b_token = token_list[bt_index]
        
        
        if f_token == '_(_)' and b_token == '엇(VerbPrefix)':          
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Exclamation)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if '(Noun)' in f_token or '(Adjective)' in f_token:
            if b_token == '쵸(VerbPrefix)':          
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Adjective)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        
        if f_token == '하(Suffix)' and '(Josa)' in b_token:      
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Josa)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token


        
        if f_token == '이(Determiner)' or f_token == '저(Determiner)' or f_token == '그(Determiner)':
            if b_token == '거(Noun)' or b_token == '걸(Noun)' or b_token == '것(Noun)' or b_token == '게(Noun)' or b_token == '고(Suffix)' or b_token == '기(Suffix)': 
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Determiner)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
                
                
        if f_token == '그(Determiner)':
            if b_token == '정도(Noun)' or b_token == '런가(Noun)' or b_token == '래야(Noun)': 
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Adverb)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        
        if f_token == '이(Determiner)':
            if b_token == '지(Suffix)' or b_token == '당(Noun)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Eomi)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        
        if f_token == '네(Determiner)' and b_token == '여(Noun)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Eomi)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
                
        
        if f_token == '이(Determiner)' and b_token == '기적(Noun)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Adjective)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
            
        if f_token == '이(Determiner)' or f_token == '그(Determiner)' :
            if b_token == '제(Suffix)' or b_token == '젠(Noun)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Adverb)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
            
            
            
        if f_token == '저(Determiner)' and b_token == '나(Noun)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Noun)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if f_token == '내(Determiner)' or f_token == '두(Determiner)':
            if b_token == '고(Modifier)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Verb)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        
        if f_token == '이(Determiner)' and b_token == '고(Modifier)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Determiner)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
            
        if f_token == '내(Determiner)':
            if b_token == '일(Suffix)' or b_token == '일(Modifier)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Noun)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
            
            
        if f_token == '이(Determiner)': 
            if b_token == '사(Modifier)' or b_token == '모(Modifier)' or b_token == '사(Suffix)' or b_token == '모(Suffix)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Noun)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
            
        if f_token == '그(Determiner)':
            if b_token == '니(Suffix)' or b_token == '니(Modifier)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Determiner)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        if f_token == '이(Determiner)' and b_token == '엿(Modifier)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Eomi)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if '(Number)' in f_token and  b_token == '퍼(PreEomi)':          
            b_token = b_token.replace('PreEomi', 'Suffix')

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token

            
        if '(Noun)' in f_token and  b_token == '두(Josa)':          
            
            f_token = f_token
            b_token = b_token.replace('Determiner', 'Suffix')

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
            
        if f_token == '너(Modifier)' and b_token == '네(Modifier)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Modifier)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
        
        
        if '(Modifier)' in f_token and len(f_token) == 11 and f_token != '너(Modifier)':
            if '(Modifier)' in b_token and len(b_token) == 11:           
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Modifier)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token  
            
            
        if f_token == '잘(VerbPrefix)' and b_token == '못(VerbPrefix)':    
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Adjective)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
            
        if '(Noun)' in f_token:
            if b_token == '적(Suffix)' or b_token == '계(Suffix)' or b_token == '급(Suffix)':
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Adjective)'
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
            
            
        if '(Noun)' in f_token and b_token == '어(Suffix)':
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

            f_token = assemble_token + '(Verb)'
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token
            
            
    ws_indexs = np.where(np.array(token_list) == '')[0].tolist()
    w = 0
    for ws in ws_indexs:
        ws -= w
        del token_list[ws]
        w += 1
    
    re_sentence = ' '.join(token_list)

    return(re_sentence)
            
            
            
def refine_pos2(tokenized_sentence):
    token_list = tokenized_sentence.split()
    
    for i in range(len(token_list)-1):
        ft_index = i
        bt_index = i+1
        
        f_token = token_list[ft_index]
        b_token = token_list[bt_index]
        
        
        if f_token == '이(Suffix)' or f_token == '중(Suffix)':
            if '(Josa)' in b_token:
                assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))

                f_token = assemble_token + '(Adverb)' 
                b_token = ''

                token_list[ft_index] = f_token
                token_list[bt_index] = b_token
        
        
        if '(Josa)' in f_token and '(Josa)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Eomi)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token


        if '(PreEomi)' in f_token and '(Eomi)' in b_token:
            assemble_token = re.sub(pattern = r'\([^)]*\)', repl='', string = str(f_token + b_token))
            
            f_token = assemble_token + '(Eomi)' 
            b_token = ''

            token_list[ft_index] = f_token
            token_list[bt_index] = b_token

        
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
            
            
    ws_indexs = np.where(np.array(token_list) == '')[0].tolist()
    w = 0
    for ws in ws_indexs:
        ws -= w
        del token_list[ws]
        w += 1
    
    re_sentence = ' '.join(token_list)

    return(re_sentence)
```

```python
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(fix_foreign)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(fix_suffix_Noun)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(restore_word)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(refine_pos1)
tokenized_df['tokenized'] = tokenized_df['tokenized'].progress_apply(refine_pos2)
tokenized_df = tokenized_df[['sex', 'age', 'contents', 'tokenized', 'token_count']]
tokenized_df
```

    100%|████████████████████████████████████████████████████████████████████████████| 3502912/3502912 [03:24<00:00, 17136.46it/s]
    ...생략...



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
      <td>나(Noun) 지금(Noun) 밥(Noun) 머거(Verb) 2시간(Number) ...</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>남성</td>
      <td>20대</td>
      <td>헐 ㅠㅠ 언넝호텔들가ㅠㅠ 엄청피건할첸데 나는인낫러요 나 두시출근이다ㅎㅎㅎㅎ 퀵으로한...</td>
      <td>헐(Verb) ㅠㅠ(KoreanParticle) 언넝(Noun) 호텔들(Noun) ...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>여성</td>
      <td>20대</td>
      <td>학생이면좋구! 왜혼자다니냐고오..... 와 내친군학교나감 ㅋㅋㅋㅋㅋ 그르네 막졸업한...</td>
      <td>학생이(Noun) 면(Josa) 좋구(Adjective) !(Punctuation)...</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남성</td>
      <td>20대</td>
      <td>훔 학생 없는데...주변에... 아니 복학하고 학교를 못가는데 어케 친구가있냐.. ...</td>
      <td>훔(Noun) 학생(Noun) 없는데(Adjective) ...(Punctuatio...</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>여성</td>
      <td>30대</td>
      <td>참나 내가뭐얼마나그랬다고 웃기는사람이야지짜 너무화난당.. 근데오빠는말을또 잘해서 내...</td>
      <td>참나(Noun) 내(Noun) 가(Josa) 뭐(Noun) 얼마나(Noun) 그랬다...</td>
      <td>63</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>3554460 rows × 5 columns</p>




```python
tokenized_df.to_csv('카톡대화_Tokenized.csv(pos 교정)', index = False)
```




<br><br>
## 03. 상대 편향도/상대 빈출도 딕셔너리 구축
### 형태소 재분류한 딕셔너리 생성




```python
re_dic = pos_token_countor(tokenized_df)
re_dic['word'], re_dic['pos'] = zip(*re_dic['Token'].apply(lambda x: word_pos_split(x)))
re_dic = re_dic[['Token', 'word', 'pos', 'Token_freq', 'Total_ratio']]
re_dic.to_csv('카톡대화_pos_dic(pos 교정).csv', index=False)
```




<br><br>
### 토큰의 상대 빈출도 및 상대 편향도 계산


```python
def feature_extractor(data, dic , col, classes):
    classes_dic = dic.copy()
    for cls in classes:
        cls_df = data.loc[data[f'{col}'] == cls]

        Freq_voca = {t:0.1 for t in dic['Token']} # 문서의 작성자가 cls일 때, 특정 토큰의 빈도 수의 총합
        Bias_voca = Freq_voca.copy() # 문서의 작성자가 cls일 때, 특정 토큰의 출현여부의 총합
        
        splited_li = split_dataframe(cls_df, size=1000)
        n = 0
        for splited_df in tqdm(splited_li, total=len(splited_li), desc=f"{cls}"): 
            if n%500 == 0:
                gc.collect()
            n+=1
            
            for index, row in splited_df.iterrows():
                token_list = row['tokenized'].split(' ') # 문서 토크나이징  

                for token in token_list: # 특정 토큰의 빈도 수 카운팅
                    if token in Freq_voca:
                        Freq_voca[token] += 1

                unique_tokens = list(set(token_list)) # 특정 토큰의 출연여부 카운팅
                for unique in unique_tokens:
                    if unique in Bias_voca:
                        Bias_voca[unique] += 1
  
        
        Freq_dic = pd.DataFrame(Freq_voca.items(), columns=['Token', f'Freq_{cls}'])
        Freq_dic[f'Freq_ratio_{cls}'] = Freq_dic[f'Freq_{cls}']/len(cls_df)
        
        Bias_dic = pd.DataFrame(Bias_voca.items(), columns=['Token', f'Bias_{cls}'])
        Bias_dic[f'Bias_ratio_{cls}'] = Bias_dic[f'Bias_{cls}']/len(cls_df) 
        
        cls_dic = pd.merge(dic[['Token']], Freq_dic, how='left', on='Token')                        
        cls_dic = pd.merge(cls_dic, Bias_dic, how='left', on='Token')
              
        classes_dic = pd.merge(classes_dic, cls_dic, how='left', on='Token')
        
    return classes_dic 
```


```python
gender_class = ['남성', '여성']
gender_dic = feature_extractor(tokenized_df, re_dic, 'sex', gender_class)

gender_dic['Gender_Freq'] = np.log(gender_dic['Freq_ratio_남성']/gender_dic['Freq_ratio_여성']) # 성별 상대 빈출도
gender_dic['Gender_Bias'] = np.log(gender_dic['Bias_ratio_남성']/gender_dic['Bias_ratio_여성']) # 성별 상대 편향도
gender_dic = gender_dic[['Token', 'word', 'pos', 'Token_freq', 'Total_ratio', 'Gender_Freq', 'Gender_Bias']]
```

    남성: 100%|██████████████████████████████████████████████████████████████████████| 791/791 [00:48<00:00, 16.28it/s]
    ...생략...




```python
age_class = ['20대 미만', '20대', '30대', '40대', '50대', '60대', '70대 이상'] 
age_dic = feature_extractor(tokenized_df, re_dic, 'age', age_class)

other_dic = token_dic[['Token']].copy().reset_index(drop=True)
for age in age_class:
    ages = ['20대 미만', '20대', '30대', '40대', '50대', '60대', '70대 이상'] 
    others_num = len(df.loc[df['age'] != age])
    ages.remove(age)
    
    other_dic[f'Freq_ratio_others_{age}'] = (age_dic[[f'Freq_{a}' for a in ages]].sum(axis=1))/others_num
    other_dic[f'Bias_ratio_others_{age}'] = (age_dic[[f'Bias_{a}' for a in ages]].sum(axis=1))/others_num
    
    age_dic[f'{age}_Freq'] = np.log(age_dic[f'Freq_ratio_{age}']/other_dic[f'Freq_ratio_others_{age}']) 
    age_dic[f'{age}_Bias'] = np.log(age_dic[f'Bias_ratio_{age}']/other_dic[f'Bias_ratio_others_{age}']) 
    
age_dic = age_dic[['Token', 
                   '20대 미만_Freq', '20대 미만_Bias', '20대_Freq', '20대_Bias',
                   '30대_Freq', '30대_Bias', '40대_Freq', '40대_Bias',
                   '50대_Freq', '50대_Bias', '60대_Freq', '60대_Bias',
                   '70대 이상_Freq', '70대 이상_Bias']]

age_dic = age_dic.rename(columns={'20대 미만_Freq':'A10_Freq', '20대 미만_Bias':'A10_Bias',
                        '20대_Freq':'A20_Freq', '20대_Bias':'A20_Bias',
                        '30대_Freq':'A30_Freq', '30대_Bias':'A30_Bias',
                        '40대_Freq':'A40_Freq', '40대_Bias':'A40_Bias',
                        '50대_Freq':'A50_Freq', '50대_Bias':'A50_Bias',
                        '60대_Freq':'A60_Freq', '60대_Bias':'A60_Bias',
                        '70대 이상_Freq':'A70_Freq', '70대 이상_Bias':'A70_Bias'})
```

    20대 미만: 100%|█████████████████████████████████████████████████████████████████| 103/103 [00:07<00:00, 14.59it/s]
    ...생략...




```python
feature_dic = pd.merge(gender_dic, age_dic, how='left', on='Token')
feature_dic
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
feature_dic.to_csv('카톡대화_feature_dic.csv', index=False)
```


