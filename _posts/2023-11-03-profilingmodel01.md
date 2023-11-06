---
layout: single
title:  "Part 01. 한국어 온라인 텍스트에 대한 탐색적 연구"
categories: Project:Profileing_model
tag: [EDA, NLP]
---
{: .notice--primary} 
1. 한국어 온라인 문체에서의 성별/연령 별 특징 분석 <br>
2. 기존 NLP 및 LLM의 한국어 자연어 처리에 대한 한계 파악/대안 제시(변수 개발)<br>

## 데이터 정보
- **출처**: [AIHUB, 한국어 SNS 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=114)
<br><br>
- **데이터 속성:** 화자 성별, 연령대, 거주지역, 대화 유형, 대화 주제, 발신 시간/순서 포함
    - **일상 대화 200만 건**
    - **파일형식:** JSON
<br><br>
- **데이터 처리**
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
    
    
    미용과건강.json 불러오기 완료(2/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 99383/99383 [00:02<00:00, 48857.42it/s]
    

    대화 세트: 99,383개
    화자 별 대화뭉치: 210,931개
    
    
    상거래(쇼핑).json 불러오기 완료(3/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 115503/115503 [00:02<00:00, 41743.76it/s]
    

    대화 세트: 115,503개
    화자 별 대화뭉치: 245,487개
    
    
    시사교육.json 불러오기 완료(4/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 89975/89975 [00:02<00:00, 38973.43it/s]
    

    대화 세트: 89,975개
    화자 별 대화뭉치: 192,916개
    
    
    식음료.json 불러오기 완료(5/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 146620/146620 [00:03<00:00, 38127.18it/s]
    

    대화 세트: 146,620개
    화자 별 대화뭉치: 310,710개
    
    
    여가생활.json 불러오기 완료(6/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 190306/190306 [00:05<00:00, 36011.40it/s]
    

    대화 세트: 190,306개
    화자 별 대화뭉치: 416,406개
    
    
    일과직업.json 불러오기 완료(7/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 107976/107976 [00:02<00:00, 45922.84it/s]
    

    대화 세트: 107,976개
    화자 별 대화뭉치: 233,041개
    
    
    주거와생활.json 불러오기 완료(8/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 197528/197528 [00:04<00:00, 40540.36it/s]
    

    대화 세트: 197,528개
    화자 별 대화뭉치: 420,132개
    
    
    행사.json 불러오기 완료(9/9)
    

    json to csv: 100%|██████████████████████████████████████████████████████████| 141205/141205 [00:03<00:00, 44828.30it/s]
    

    대화 세트: 141,205개
    화자 별 대화뭉치: 305,369개
    
    
    


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
    
    
    미용과건강.json 불러오기 완료(2/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 12423/12423 [00:00<00:00, 45924.05it/s]
    

    대화 세트: 12,423개
    화자 별 대화뭉치: 26,410개
    
    
    상거래(쇼핑).json 불러오기 완료(3/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 14439/14439 [00:00<00:00, 39023.13it/s]
    

    대화 세트: 14,439개
    화자 별 대화뭉치: 30,713개
    
    
    시사교육.json 불러오기 완료(4/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 11247/11247 [00:00<00:00, 48714.92it/s]
    

    대화 세트: 11,247개
    화자 별 대화뭉치: 24,133개
    
    
    식음료.json 불러오기 완료(5/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 18328/18328 [00:00<00:00, 50465.28it/s]
    

    대화 세트: 18,328개
    화자 별 대화뭉치: 38,876개
    
    
    여가생활.json 불러오기 완료(6/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 23789/23789 [00:00<00:00, 49149.45it/s]
    

    대화 세트: 23,789개
    화자 별 대화뭉치: 52,068개
    
    
    일과직업.json 불러오기 완료(7/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 13498/13498 [00:00<00:00, 47982.39it/s]
    

    대화 세트: 13,498개
    화자 별 대화뭉치: 29,072개
    
    
    주거와생활.json 불러오기 완료(8/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 24691/24691 [00:00<00:00, 40866.16it/s]
    

    대화 세트: 24,691개
    화자 별 대화뭉치: 52,572개
    
    
    행사.json 불러오기 완료(9/9)
    

    json to csv: 100%|████████████████████████████████████████████████████████████| 17651/17651 [00:00<00:00, 40031.52it/s]
    

    대화 세트: 17,651개
    화자 별 대화뭉치: 38,245개
    
    
    


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
```

    병합 진행률: 100%|███████████████████████████████████████████████████████████████████████| 9/9 [00:32<00:00,  3.59s/it]
    


```python
topic_list = ['개인및관계', '미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']

full_df = pd.DataFrame()
for topic in tqdm(topic_list, total = len(topic_list), desc='병합 진행률'):  
    topic_df = pd.read_csv(f'{topic}.csv')

    full_df = pd.concat([full_df, topic_df], axis= 0, ignore_index= True)
    full_df.to_csv('SNS_FULL_Dataset(raw).csv', index=False)
```

    병합 진행률: 100%|███████████████████████████████████████████████████████████████████████| 9/9 [01:35<00:00, 10.61s/it]
    


```python
full_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>3844730 rows × 6 columns</p>
</div>




```python
full_df = full_df.drop_duplicates(['sex', 'age', 'contents']) # 중복된 데이터 제거
full_df = full_df.dropna()
full_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




```python
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


plt.figure(figsize=(5,3))
plt.rc('font', size=20)

print('평균', full_df['length'].mean())
print('상위1%', full_df['length'].quantile(0.05))
print('하위1%', full_df['length'].quantile(0.95))

sns.distplot(full_df['length'])
plt.show()
```

    평균 80.99197912758542
    상위1% 21.0
    하위1% 173.0
    

    C:\Users\hakyeong\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_8_2.png)
    



```python
length_df = full_df.loc[full_df['length'] >=20] 
length_df = length_df.loc[length_df['length'] <=200] 
length_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>3564117 rows × 6 columns</p>
</div>




```python
# 중복된 메시지 확인
dup_df = length_df[length_df['contents'].duplicated(keep=False)].sort_values('contents')
dup_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
raw_df.to_csv('SNS_FULL_Dataset(raw_중복된 메세지 제거).csv', index=False)
```


```python

```
(SNS 200만).md…]()
