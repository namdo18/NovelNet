</br>

# 웹소설 추천을 위한 반복 소비 행동 모델링
Li et al.2022[[1](#reference)] 논문 구현 프로젝트  
`python==3.10.6`, `pytorch==1.11.0`

### 주요 사용 알고리즘
- `biGRU`, `BahdanauAttention`, `PointerNetwork`, `pointwiseLoss`


### 논문 리뷰 요약
- 논문 선정 이유
    - 2022년 9월 나온 논문으로 최신 연구경향을 담고 있을 것으로 기대
    - 주요 저자들이 텐센트 소속으로 비즈니스 목적의 실용적인 연구가 진행되었을 것으로 기대
- 핵심 내용 
    - 이용자들의 서비스 이용 패턴에서 특징값을 추출. 모델의 입력으로 이용하여 도메인 특화 추천시스템 구현 가능(인코딩 후 병합하여 시퀀스화)
    - 이용자들 대부분이 반복 소비 패턴을 보임에 따라, 이전 소비 내역을 추천하는 것이 중요 
    - 신규 이용자들은 선호 작품을 찾는 것이 친숙하지 않기 때문에, 그들을 위한 추천 시스템 필요(세션 기반 추천과 신규 이용자를 위한 추천은 동일)
    - 이용 패턴 및 반복 소비를 고려해 신규이용자들에게 웹소설을 추천해주는 세션 기반 추천시스템 `NovelNet` 제안
    - 앱 환경에서 모달을 띄워 한 개의 작품을 추천하는 경우를 가정, recall@1 을 평가지표로 `NovelNet` 0.4702 성능 달성
        - `GRU4Rec`[[2](#reference)] 0.4220, `RecentNovel` 0.4448 대비 우수한 성능 기록
        - `RecentNovel` 은 직전 소비한 소설을 추천하는 규칙 기반 알고리즘
    

### 핵심 알고리즘
- `biGRU` 를 통한 세션 정보 시퀀스 인코딩
    - 특징값 별로 개별 임베딩 레이어를 통해 벡터화
    - 세션 단위로 임베딩 값 병합 후 `biGRU` 레이어를 통해 시퀀스 인코딩
        - 인코딩 시퀀스 $S = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L], \quad \mathbf{h}_l = [\mathbf{\overset{\rightarrow}{h}}_l; \mathbf{\overset{\leftarrow}{h}}_l]$ 
        - 각 방향별 마지막 히든값 병합, $H = [\mathbf{\overset{\rightarrow}{h}}_L ; \mathbf{\overset{\leftarrow}{h}}_1]$

- `BahdanauAttention` 를 통한 새로운 소설에 대한 추천 확률 도출
    - 시퀀스 정보 압축, $\text{score}(S, H) = \text{tanh}(S \cdot W_s + H \cdot W_h) \cdot V,\quad W \in \mathbb{R}^{|\mathbf{h}| \times |\mathbf{h}|}, V \in \mathbb{R}^{|\mathbf{h}|}$
    - 대표 시퀀스 $r^S \in \mathbb{R}^{|batch| \times |\mathbf{h}|}$ 인코딩, $\text{value}(\text{score}, S) = \text{softmax}(\text{score}^T)S$
    - 소설임베딩 값과 유사도 계산 및 마스킹을 통한 새로운 소설에 대한 추천 확률 도출, $p^{new} = \text{softmax}(r^SW^T_{\text{emb}} \circledcirc m^c),\quad c=\text{consumed}$


- `PointerNetwork` 를 통한 이전 소비 내역에 대한 추천 확률 도출
    - 시퀀스 정보 압축, $\text{score}(S, H) = \text{tanh}(S \cdot W^c_s + H \cdot W^c_h) \cdot V^c,\quad W^c \in \mathbb{R}^{|\mathbf{h}| \times |\mathbf{h}|}, V^c \in \mathbb{R}^{|\mathbf{h}|}$
    - 빈도 수를 고려한 소설별 확률값 도출, $p^{listwise} = \text{softmax}(\text{score}) \cdot E_{\text{novel}},\quad E = \text{1 of N encoding}$
    - 인터렉션별 확률 도출, $p^{pointwise} = \text{sigmoid}(\text{score})$
        - $p^{listwise}$ 에 대해 세션 길이가 1인 경우, 실제 소비 의도와 상관 없이 하나의 소설에 대한 확률이 1로 설정되는 것을 방지하기 위해 사용
        - 실제 추천 시에는 $p^{pointwise} \cdot E_{\text{novel}}$ 이용( $p^{new}$ 와 비교하여 가장 높은 확률 소설 추천)

- `pointwiseLoss` 및 손실함수 설정 $\mathcal{L} = \mathcal{L} + \mathcal{L}_{pointwise}$
    - $\mathcal{L} = \text{NLL-loss}(p), \quad p = p^{listwise} \ \text{if} \ y \in N^c \ \text{else} \ p^{new}$
    - $\mathcal{L}_{pointwise} = \lambda \cdot \text{BCE}(m \circledcirc p^{pointwise})$
        - $\lambda$ 를 통해 `pointwiseLoss` 에 가중치를 부여하여 학습
        - $m$ 은 마지막 인터렉션(세션 정보 누적)에 대해서만 연산할 수 있도록 하는 마스크 
        

### 프로젝트 수행결과

- 텐센트 QQ브라우저의 웹소설 이용 데이터 이용
    - 2021.11.11~22 기간에서 랜덤샘플링 된 세션 중 1000 개의 세션에 대한 인터렉션 데이터들을 학습 데이터셋으로 이용
    - 2021.11.23~24 기간에서 랜덤샘플링 된 세션 중 300 개의 세션에 대한 인터렉션 데이터들을 테스트 데이터셋으로 이용
    
- 알고리즘 구현 및 효과성 검증 실험 결과
    - 총 2,503 명의 신규 이용자들 각각에 대해, 2,760 개의 소설 중 다음으로 읽을 소설을 정확히(recall@1) 예측할 확률
    - 0.5825 의 성능 기록(경우의 수가 줄어들면서 성능향상 추정)
    
- 추가 실험 진행
    - $\text{CrossEntropy}(\text{softmax}((1-\alpha) \cdot p^{new} + \alpha \cdot p^{listwise}))$ 를 통해 학습
        - 이전 소비 이력에 대한 가중치 $\alpha$ 를 고려하여 밸런싱한 확률 기반 학습 진행 
        - 이전 소비 이력에 대한 추천이 성능에 미치는 영향 조사 목적(짧은 세션 오류 감안)
        - 극단적 상황 $\alpha = 0.9$ 상황에서도 오히려 모델 성능은 0.7013 으로 +0.1188p 향상
        
- 프로젝트 결과
    - `RecentNovel` 의 성능이 0.4448[[1](#reference)] 이라는 것은 직전에 읽었던 소설을 바로 다시 읽은 데이터가 44.48% 에 달한다는 것
        - `NovelNet` 이 이러한 데이터들을 학습하여 잘 추천해도, 이용자들 입장에서 직전에 읽은 소설이 추천(모달)되는 것은 바로가기 기능에 불과할 것
    - 추가 실험 진행 결과, 사실상 이전 소비 내역만 고려했을 때의 recall@1 성능이 70% 수준
    - 위와 같은 상황에서 학습된 모델의 성능은 이용자들에게 다양한 콘텐츠 소비를 유도하고자 하는 추천시스템의 목적[[3](#reference)]에 부합하지 못할 것
    - 또한, Li et al.2022[[1](#reference)] 에서는 실제로 유효 소비라고 볼 수 없는 소설들에 대해서도 추천하도록 학습이 진행됨. 이에 따라 실효성을 기대하기 어려울 것.
    - 추천 시스템 목적[[3](#reference)] 에 부합하며, 실효성을 충족시킬 수 있는 모델 개발 필요

</br>

## Reference
###### [1]Yuncong Li, et al, 2022, Modeling User Repeat Consumption Behavior for Online Novel Recommendation, RecSys’22 September 18–23
###### [2]Balazs Hidasi, et al, 2016, Session-Based Recommendations with Recurrent Neural Networks, ICLR 2016
###### [3]김대원, 2019, 추천 알고리즘의 개념과 적용 그리고 발전의 양상, Broadcasting Trend & Insight October 2019 Vol.20, 한국콘텐츠 진흥원




