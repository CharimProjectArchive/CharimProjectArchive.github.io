---
layout: single
title:  "Part 00. 온라인 메시지 작성자 프로파일링 모델 개발: 데이터셋 기초 탐색"
categories: Project:Profileing_model
tag: [EDA, NLP]
---
{: .notice--primary} 
💡**프로젝트 배경**<br>

개인정보 보호에 대한 사회적 분위기에 따라 구글 써드파티 제공 중단, 애플 사용자 정보 공개 중단 등 사용자 정보를 수집하는 것이 어려워지고 있음. 자사 플랫폼에 가입한 사용자 외 고객 데이터를 얻는 것은 더욱 어려움.<br><br> 
맥락정보를 활용한 고객 분석 및 타겟팅 전략에 대한 관심이 높아지고 있으나 대표적으로 고객이 생산하는 맥락정보인 채팅, 리뷰, 피드 등으로 파악할 수 있는 정보는 매우 제한적임. 특히, 사용자의 인구통계적 정보가 제공되는 경우는 매우 드물어 고객 분석 및 타겟팅에 활용할 수 없음.<br><br> 
**채팅, 리뷰, 피드 등에서 사용자의 인구 통계적 정보를 추정할 수 있다면, 고객 분석 및 타겟팅에 활용할 수 있을 뿐만아니라 챗봇/메타버스 서비스/CRM 서비스 등을 고도화시킬 수 있을 것으로 기대됨**<br><br>

{: .notice--primary} 
🎯**프로젝트 목적**<br>

고객이 온라인 상에서 생산하는 맥락정보(채팅, 리뷰, 피드 등의 텍스트 정보)에서 작성자의 인구 통계적 정보(성별/연령)를 추정하는 작성자 프로파일링 모델 개발<br><br>


## 기초 탐색 요약<br>
- 어미 변형, 발음 그대로의 표현, 오타 등으로 동일한 어간/의미를 갖는 표현이지만 변형된 표현이 상당히 많음
- 자음 또는 모음으로 구성된 불완전한 표현, 문장기호, 특수기호, 이모지(이하 특수표현)가 메시지 대부분에 포함되어 있음<br>
⇒ 일반적인 불용어 제거 및 표제화에서 제거되는 표현<br><br>

**인사이트/가설**
- 변형된 표현 및 특수표현 자체가 작성자의 인구 통계적 특성(성별/연령)과 중요한 상관관계를 가질 수 있음
1. Target feature(성별/연령)과 표현의 변형/특수표현 간의 상관관계 파악이 필요
   - 상관관계 분석을 위한 변수 개발 필요

2. 변형된 표현 및 특수한 표현이 제거되지 않도록 전처리 수행
<br><br>

## 데이터 정보
- **출처**: [AIHUB, 한국어 SNS 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=114)
<br><br>
- **데이터 속성:** 화자 성별, 연령대, 거주지역, 대화 유형, 대화 주제, 발신 시간/순서 포함
    - **일상 대화 200만 건**
    - **파일형식:** JSON
<br>

## 데이터 처리
- json 파일을 dataframe 형식으로 변환
- 2인 이상의 대화 데이터(대화뭉치)에서 발화자 id를 기준으로 메시지 병합
- 메타 데이터 중 인구통계적 정보만 추출(성별, 연령, 거주지)


```python
import pandas as pd
import json
from pprint import pprint
from glob import glob
from tqdm import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
```


```python
#경로 내 json 파일 이름을 리스트로 변환
file_list = glob('./한국어 SNS/Training./*.json')

file_num = 0
load_failed = []
for file in file_list:     #리스트의 json 파일 선택
    file_num += 1
    file_name = file.split('\\')[-1]
    print("{} 불러오는 중...".format(file_name), end = '\r')
    
    
    try:     #json 파일 열기
        with open(file, 'rt', encoding = 'UTF-8') as f:
            data = json.load(f)['data']
        print("{} 불러오기 완료({}/{})".format(file_name, file_num, len(file_list)))

    except:     #메모리 부족으로 파일 열기 실패
        load_failed.append(file_name)
        print("{} 불러오기 실패: 메모리 부족\n".format(file_name))
        pass
    
    else:  #정상적으로 파일을 열었을 때
        total_data = []
        for log in tqdm(data, total=len(data), desc="json to csv", mininterval=0.1):     #진행 경과를 바로 표시
            dialogueID = log['header']['dialogueInfo']['dialogueID']
            participantsInfo = log['header']['participantsInfo']
            dialogue = log['body']

            #대화 참여자 정보
            participant_info = []
            for participant in participantsInfo:
                participantID = participant['participantID']
                gender = participant['gender']
                age = participant['age']
                resident = participant['residentialProvince']

                participant = [participantID, gender, age, resident]
                participant_info.append(participant)



            #메세지 정보
            speaker_id = []
            speaker_text = []
            for message in dialogue: 
                text_id = message['participantID']
                speaker_id.append(text_id) 

                text = message['utterance']
                speaker_text.append(text) 

            unique_id = list(set(speaker_id)) #고유 speacker 확인
            unique_id.sort()

            text_by_speaker= []
            
            c = 0
            for i in unique_id:    #동일 화자 메세지의 인덱스 추출
                idx = list(filter(lambda x: speaker_id[x] == i, range(len(speaker_id))))

                same_id_text = []
                for n in idx:    #동일 화자 메세지 병합 및 불필요 패턴 제거
                    same_id_text.append(speaker_text[n])
                same_id_conversation = ' '.join(same_id_text)
                same_id_conversation = same_id_conversation.replace('#@이름#', 'User').replace('#@계정#', 'UserID').replace('#@신원#', '').replace('#@전번#', '').replace('#@주소#', '').replace('#@번호#', '').replace('#@소속#', '').replace('#@금융#', '').replace('#@이모티콘#', '').replace('#@시스템#사진#', '').replace('#@시스템#검색#', '').replace('#@시스템#송금#', '').replace('#@시스템#동영상# ', '').replace('#@기타#', '').replace('#@URL#', '')

                speaker = participant_info[c]
                
                #고유 화자를 기준으로 한 셀값 정렬
                full_message_by_id = [dialogueID] + speaker + [same_id_conversation]
                total_data.append(full_message_by_id)
                c += 1

        df = pd.DataFrame(total_data, columns=['dialogueID', 'speakerID', 'sex',  'age', 'resident', 'contents'])
        
        #csv로 저장
        df.to_csv("Training_{}.csv".format(file_name.replace('.json', '')), index=False)
        print('대화 세트: {0:,}개'.format(len(data)))
        print('화자 별 대화뭉치: {0:,}개'.format(df.shape[0]))
        print('\n')
```

    개인및관계.json 불러오기 완료(1/9)
    json to csv: 100%|██████████████████████████████████████████████████████████| 511496/511496 [00:12<00:00, 40772.61it/s]
    대화 세트: 511,496개
    화자 별 대화뭉치: 1,082,221개
    
    (...생략...)




```python
#경로 내 json 파일 이름을 리스트로 변환
file_list = glob('./한국어 SNS/Validation/*.json')

file_num = 0
load_failed = []
for file in file_list:     #리스트의 json 파일 선택
    file_num += 1
    file_name = file.split('\\')[-1]
    print("{} 불러오는 중...".format(file_name), end = '\r')
    
    
    try:     #json 파일 열기
        with open(file, 'rt', encoding = 'UTF-8') as f:
            data = json.load(f)['data']
        print("{} 불러오기 완료({}/{})".format(file_name, file_num, len(file_list)))

    except:     #메모리 부족으로 파일 열기 실패
        load_failed.append(file_name)
        print("{} 불러오기 실패: 메모리 부족\n".format(file_name))
        pass
    
    else:  #정상적으로 파일을 열었을 때
        total_data = []
        for log in tqdm(data, total=len(data), desc="json to csv", mininterval=0.1):     #진행 경과를 바로 표시
            dialogueID = log['header']['dialogueInfo']['dialogueID']
            participantsInfo = log['header']['participantsInfo']
            dialogue = log['body']

            #대화 참여자 정보
            participant_info = []
            for participant in participantsInfo:
                participantID = participant['participantID']
                gender = participant['gender']
                age = participant['age']
                resident = participant['residentialProvince']

                participant = [participantID, gender, age, resident]
                participant_info.append(participant)



            #메세지 정보
            speaker_id = []
            speaker_text = []
            for message in dialogue: 
                text_id = message['participantID']
                speaker_id.append(text_id) 

                text = message['utterance']
                speaker_text.append(text) 

            unique_id = list(set(speaker_id)) #고유 speacker 확인
            unique_id.sort()

            text_by_speaker= []
            
            c = 0
            for i in unique_id:    #동일 화자 메세지의 인덱스 추출
                idx = list(filter(lambda x: speaker_id[x] == i, range(len(speaker_id))))

                same_id_text = []
                for n in idx:    #동일 화자 메세지 병합 및 불필요 패턴 제거
                    same_id_text.append(speaker_text[n])
                same_id_conversation = ' '.join(same_id_text)
                same_id_conversation = same_id_conversation.replace('#@이름#', 'User').replace('#@계정#', 'UserID').replace('#@신원#', '').replace('#@전번#', '').replace('#@주소#', '').replace('#@번호#', '').replace('#@소속#', '').replace('#@금융#', '').replace('#@이모티콘#', '').replace('#@시스템#사진#', '').replace('#@시스템#검색#', '').replace('#@시스템#송금#', '').replace('#@시스템#동영상# ', '').replace('#@기타#', '').replace('#@URL#', '')

                speaker = participant_info[c]
                
                #고유 화자를 기준으로 한 셀값 정렬
                full_message_by_id = [dialogueID] + speaker + [same_id_conversation]
                total_data.append(full_message_by_id)
                c += 1

        df = pd.DataFrame(total_data, columns=['dialogueID', 'speakerID', 'sex',  'age', 'resident', 'contents'])
        
        #csv로 저장
        df.to_csv("Validation_{}.csv".format(file_name.replace('.json', '')), index=False)
        print('대화 세트: {0:,}개'.format(len(data)))
        print('화자 별 대화뭉치: {0:,}개'.format(df.shape[0]))
        print('\n')
```

    개인및관계.json 불러오기 완료(1/9)
    json to csv: 100%|████████████████████████████████████████████████████████████| 63938/63938 [00:01<00:00, 42361.49it/s]
    대화 세트: 63,938개
    화자 별 대화뭉치: 135,428개
    
    (...생략...)
    
    
    


```python
topic_list = ['개인및관계', '미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']
for topic in tqdm(topic_list, total = len(topic_list), desc='병합 진행률'):  
    train = pd.read_csv(f'Training_{topic}.csv')
    valid = pd.read_csv(f'Validation_{topic}.csv')
    
    train = train.astype({'contents':'str'})
    train['length'] = train['contents'].apply(lambda x: len(x))
    
    valid = valid.astype({'contents':'str'})
    valid['length'] = valid['contents'].apply(lambda x: len(x))
    
    topic_df = pd.concat([train, valid], axis= 0, ignore_index= True)
    topic_df['topic'] = topic
    topic_df = topic_df[['topic', 'sex', 'age', 'resident', 'contents', 'length']]
    topic_df.to_csv(f'./{topic}.csv', index=False)

topic_list = ['개인및관계', '미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']

full_df = pd.DataFrame()
for topic in tqdm(topic_list, total = len(topic_list), desc='병합 진행률'):  
    topic_df = pd.read_csv(f'{topic}.csv')

    full_df = pd.concat([full_df, topic_df], axis= 0, ignore_index= True)
    full_df.to_csv('SNS_FULL_Dataset(raw).csv', index=False)
```

    병합 진행률: 100%|███████████████████████████████████████████████████████████████████████| 9/9 [01:35<00:00, 10.61s/it]
    병합 진행률: 100%|███████████████████████████████████████████████████████████████████████| 9/9 [01:35<00:00, 10.61s/it]
    



```python
full_df = full_df.drop_duplicates(['sex', 'age', 'contents']) # 중복된 데이터 제거
full_df = full_df.dropna()
full_df
```




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
    </tr>
    <tr>
      <th>1</th>
      <td>개인및관계</td>
      <td>남성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>헐 ㅠㅠ 언넝호텔들가ㅠㅠ 엄청피건할첸데 나는인낫러요 나 두시출근이다ㅎㅎㅎㅎ 퀵으로한...</td>
      <td>130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>학생이면좋구! 왜혼자다니냐고오..... 와 내친군학교나감 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 그르네 ...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>개인및관계</td>
      <td>남성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>훔 학생 없는데...주변에... 아니 복학하고 학교를 못가는데 어케 친구가있냐.. ...</td>
      <td>74</td>
    </tr>
    <tr>
      <th>4</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>30대</td>
      <td>충청북도</td>
      <td>참나 내가뭐얼마나그랬다고 웃기는사람이야지짜 너무화난당.. 근데오빠는말을또 잘해서 내...</td>
      <td>146</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3844725</th>
      <td>행사</td>
      <td>여성</td>
      <td>20대</td>
      <td>서울특별시</td>
      <td>맞앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 하루로 되냐곸ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ...</td>
      <td>57</td>
    </tr>
    <tr>
      <th>3844726</th>
      <td>행사</td>
      <td>여성</td>
      <td>30대</td>
      <td>경기도</td>
      <td>다음날은 일어나서 근처 또 유명빵집 가서 점심을 먹고 집으로 복귀! 어제 점심먹을곳...</td>
      <td>61</td>
    </tr>
    <tr>
      <th>3844727</th>
      <td>행사</td>
      <td>남성</td>
      <td>20대</td>
      <td>인천광역시</td>
      <td>집을 가는군여! 아주우!!!!! 좋습네다 User!!!! 아주좋아요! 다음주야 어차피!</td>
      <td>48</td>
    </tr>
    <tr>
      <th>3844728</th>
      <td>행사</td>
      <td>여성</td>
      <td>20대</td>
      <td>서울특별시</td>
      <td>근데 동창회 인문계까지 싹 다 하려면 절대 못해.. 진짜 망상만 가능 그거 수용해줄...</td>
      <td>121</td>
    </tr>
    <tr>
      <th>3844729</th>
      <td>행사</td>
      <td>여성</td>
      <td>20대</td>
      <td>서울특별시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅠㅠ엉엉 된다고 해주세요 그니까 망상 동창회 그정도면 연회장 빌려야함 ...</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
<p>3835493 rows × 6 columns</p>
</div>
<br><br>

## 기초 탐색
- 문장 길이, feautre 별 분포 확인



```python
# 문장 길이 확인 및 이상치 제거
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(5,3))
plt.rc('font', size=20)

print('평균', full_df['length'].mean())
print('상위5%', full_df['length'].quantile(0.05))
print('하위5%', full_df['length'].quantile(0.95))

sns.histplot(full_df['length'])
plt.show()
```

    평균 80.99197912758542
    상위5% 21.0
    하위5% 173.0

    


    
![png](../image/output_8_2.png)
    



```python
length_df = full_df.loc[full_df['length'] >=20] 
length_df = length_df.loc[length_df['length'] <=200] 
```




```python
# feature는 다르지만 동일한한 메시지 확인
dup_df = length_df[length_df['contents'].duplicated(keep=False)].sort_values('contents')
dup_df
```




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1853440</th>
      <td>시사교육</td>
      <td>여성</td>
      <td>20대 미만</td>
      <td>인천광역시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2307313</th>
      <td>여가생활</td>
      <td>여성</td>
      <td>30대</td>
      <td>서울특별시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>25</td>
    </tr>
    <tr>
      <th>153266</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>30대</td>
      <td>경기도</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>20</td>
    </tr>
    <tr>
      <th>136342</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>충청남도</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2073386</th>
      <td>식음료</td>
      <td>남성</td>
      <td>20대</td>
      <td>충청북도</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1107375</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>서울특별시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2425483</th>
      <td>여가생활</td>
      <td>여성</td>
      <td>30대</td>
      <td>서울특별시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ...</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1527440</th>
      <td>상거래(쇼핑)</td>
      <td>여성</td>
      <td>20대</td>
      <td>서울특별시</td>
      <td>ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ...</td>
      <td>96</td>
    </tr>
    <tr>
      <th>670683</th>
      <td>개인및관계</td>
      <td>남성</td>
      <td>30대</td>
      <td>경기도</td>
      <td>아ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>21</td>
    </tr>
    <tr>
      <th>70518</th>
      <td>개인및관계</td>
      <td>여성</td>
      <td>20대</td>
      <td>경기도</td>
      <td>아ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 6 columns</p>
</div>




```python
dup_df.to_excel('중복 메세지 유형 확인.xlsx', index=False)
```

    


```python
# feature 별 분포 확인
m_df = df.loc[df['sex'] == '남성']
f_df = df.loc[df['sex'] == '여성']

a10_df = df.loc[df['age'] == '20대 미만']
a20_df = df.loc[df['age'] == '20대']
a30_df = df.loc[df['age'] == '30대']
a40_df = df.loc[df['age'] == '40대']
a50_df = df.loc[df['age'] == '50대']
a60_df = df.loc[df['age'] == '60대']
a70_df = df.loc[df['age'] == '70대 이상']

values = [total, male, female, a10, a20, a30, a40, a50, a60, a70]
labels = ['total', 'male', 'female', 'age10', 'age20', 'age30', 'age40', 'age50', 'age60', 'age70']

plt.figure(figsize = (20, 10))
plt.grid(True, axis = 'y')
bar = plt.bar(labels, values, color='lightgray')
plt.rcParams['font.size'] = 20
plt.ylim(0, 4000000)

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:,}\n({height/total*100:.2f}%)', ha='center', va='bottom', size = 15)

plt.show()
```


    
![png](output_4_0.png)
    



- target에 해당하는 feature의 분포가 매우 언밸런스함 ⇒ 스캐일링 고려




```python
# topic value 분포 확인
import scipy.stats as st

topic_list = ['개인및관계', '미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']

rows = []
for topic in tqdm(topic_list, total = len(topic_list)):  
    df_t = df.loc[df['topic'] == f'{topic}']
    file = topic
    total = len(df_t)
    length99 = st.t.interval(0.99, len(df_t['length'])-1, loc=np.mean(df_t['length']), scale=st.sem(df_t['length']))[1]
    male = len(df_t.loc[df_t['sex']== '남성'])
    female = len(df_t.loc[df_t['sex']== '여성'])
    a10 = len(df_t.loc[df_t['age']== '20대 미만'])
    a20 = len(df_t.loc[df_t['age']== '20대'])
    a30 = len(df_t.loc[df_t['age']== '30대'])
    a40 = len(df_t.loc[df_t['age']== '40대'])
    a50 = len(df_t.loc[df_t['age']== '50대'])
    a60 = len(df_t.loc[df_t['age']== '60대'])
    a70 = len(df_t.loc[df_t['age']== '70대 이상'])
    
    row = [file, total, length99, male, female, a10, a20, a30, a40, a50, a60, a70]
    rows.append(row)

    
cols = ['topic', 'total', 'length99', 'male', 'female', 'age10', 'age20', 'age30', 'age40', 'age50', 'age60', 'age70'] 
balance_df = pd.DataFrame(rows, columns=cols)
balance_df
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:05<00:00,  1.57it/s]
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
      <th>total</th>
      <th>length99</th>
      <th>male</th>
      <th>female</th>
      <th>age10</th>
      <th>age20</th>
      <th>age30</th>
      <th>age40</th>
      <th>age50</th>
      <th>age60</th>
      <th>age70</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개인및관계</td>
      <td>1134029</td>
      <td>77.974638</td>
      <td>292596</td>
      <td>841433</td>
      <td>41984</td>
      <td>859833</td>
      <td>200145</td>
      <td>18890</td>
      <td>10168</td>
      <td>2834</td>
      <td>175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>미용과건강</td>
      <td>221514</td>
      <td>78.514982</td>
      <td>40737</td>
      <td>180777</td>
      <td>4715</td>
      <td>160611</td>
      <td>47958</td>
      <td>3893</td>
      <td>3510</td>
      <td>766</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>상거래(쇼핑)</td>
      <td>255671</td>
      <td>78.690426</td>
      <td>43766</td>
      <td>211905</td>
      <td>3983</td>
      <td>179996</td>
      <td>59764</td>
      <td>6383</td>
      <td>4476</td>
      <td>1001</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>시사교육</td>
      <td>198409</td>
      <td>83.080023</td>
      <td>47544</td>
      <td>150865</td>
      <td>14134</td>
      <td>154678</td>
      <td>24230</td>
      <td>2710</td>
      <td>1697</td>
      <td>895</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>식음료</td>
      <td>329375</td>
      <td>74.298464</td>
      <td>66110</td>
      <td>263265</td>
      <td>5480</td>
      <td>238580</td>
      <td>71768</td>
      <td>5979</td>
      <td>6623</td>
      <td>886</td>
      <td>59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>여가생활</td>
      <td>431670</td>
      <td>78.421059</td>
      <td>84485</td>
      <td>347185</td>
      <td>11821</td>
      <td>334418</td>
      <td>77398</td>
      <td>3114</td>
      <td>2955</td>
      <td>1910</td>
      <td>54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>일과직업</td>
      <td>236247</td>
      <td>85.073682</td>
      <td>46887</td>
      <td>189360</td>
      <td>2953</td>
      <td>173194</td>
      <td>53600</td>
      <td>4030</td>
      <td>1611</td>
      <td>748</td>
      <td>111</td>
    </tr>
    <tr>
      <th>7</th>
      <td>주거와생활</td>
      <td>438561</td>
      <td>78.515856</td>
      <td>97589</td>
      <td>340972</td>
      <td>8408</td>
      <td>300678</td>
      <td>107526</td>
      <td>11770</td>
      <td>7772</td>
      <td>2235</td>
      <td>172</td>
    </tr>
    <tr>
      <th>8</th>
      <td>행사</td>
      <td>318566</td>
      <td>77.673101</td>
      <td>73144</td>
      <td>245422</td>
      <td>9229</td>
      <td>231402</td>
      <td>64480</td>
      <td>8802</td>
      <td>3915</td>
      <td>623</td>
      <td>115</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import gridspec

w = 0.1
nrow = 9# 행의 갯수
idx = np.arange(nrow)

a10 = balance_df['age10']
a20 = balance_df['age20']
a30 = balance_df['age30']
a40 = balance_df['age40']
a50 = balance_df['age50']
a60 = balance_df['age60']
a70 = balance_df['age70']

fig = plt.figure(figsize=(20, 10)) 
gs = gridspec.GridSpec(nrows=2, 
                       ncols=1, 
                       height_ratios=[3,2], 
                       width_ratios=[10])

ax1 = plt.subplot(gs[0])
ax1.bar(idx - w*3, a10, width = w)
ax1.bar(idx - w*2, a20, width = w)
ax1.bar(idx - w, a30, width = w)
ax1.bar(idx, a40, width = w)
ax1.bar(idx + w, a50, width = w)
ax1.bar(idx + w*2, a60, width = w)
ax1.bar(idx + w*3, a70, width = w)


ax2 = plt.subplot(gs[1])
ax2.bar(idx - w*3, a10, width = w)
ax2.bar(idx - w*2, a20, width = w)
ax2.bar(idx - w, a30, width = w)
ax2.bar(idx, a40, width = w)
ax2.bar(idx + w, a50, width = w)
ax2.bar(idx + w*2, a60, width = w)
ax2.bar(idx + w*3, a70, width = w)


ax1.yaxis.grid()
ax2.yaxis.grid()

ax1.set_ylim(10000, 1000000)
ax1.set_yticks([10000, 100000, 200000, 300000, 400000, 1000000])
ax1.set_ylabel("")
ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.set_xlabel("")
ax1.legend(balance_df.columns[-7:], ncol = 7) 

ax2.set_ylim(0, 10000)
ax2.set_yticks([1000, 2000, 4000, 6000, 8000, 10000])
ax2.set_ylabel("")
ax2.spines['top'].set_visible(False)

d = .7    
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

plt.xticks(idx, topic_list, rotation = 0)
plt.show()
```


    
![png](output_6_0.png)




- topic 값에 대한 분포는 상당히 언밸런스함
- 일상적인 카톡대화에서 발화되는 주제별 비율과 비슷할 것으로 보여짐




## 데이터 육안 스크리닝 
- 동일한 어간/의미를 갖지만 어미, 줄임말, 오타 등으로 변형된 표현이 매우 많은 것으로 보여짐
- 자음 또는 모음으로 구성된 불완전 표현(Korean Particle), 문장기호, 특수기호, 이모지 표현이 대부분 사용됨
    - 자음 또는 모음으로 구성된 불완전 표현: ㅋㅋㅋㅋ, ㅠㅠ, ㄹㅇ, ㅋㅑㅋㅑㅋㅑ 등
    - 문장기호 및 특수기호: . , ; $ ^^ ^-^ 등
    - 이모지: 🤣, 👍, 🙏 등
- topic value에 대한 분류기준이 적절하지 않은 것으로 보여 topic은 feature로 사용하지 않는 것으로 함




```python
raw_df.to_csv('SNS_FULL_Dataset(raw_중복된 메세지 제거).csv', index=False)
```
